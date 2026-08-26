import numpy as np
import json
import os
import glob
import sys
import re

def find_ground_truth_file(directory='.'):
    """
    Automatically find the txt file named with pure numbers in the current directory as the ground truth.
    e.g., 1450.txt, 1451.txt, etc.
    """
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    for f in txt_files:
        basename = os.path.splitext(os.path.basename(f))[0]
        if basename.isdigit():
            return f
    return None

def load_matrix_from_file(file_path):
    """
    Load matrix data from file, supporting JSON format and standard matrix format.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Try to parse JSON format
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # Support multiple key formats
            matrix = None
            if "Reconstructed Matrices" in data:
                matrix = data["Reconstructed Matrices"]
            elif "Reconstructed_Matrices" in data:
                matrix = data["Reconstructed_Matrices"]
            elif "reconstructed_matrix" in data:
                matrix = data["reconstructed_matrix"]
            
            if matrix is not None:
                result = np.rint(np.array(matrix, dtype=float)).astype(int)
                print(f"Loaded from JSON format: {file_path}")
                if result.shape[0] > 64:
                    print(f"Warning: File {file_path} has {result.shape[0]} rows, only using the first 64 rows.")
                    result = result[:64, :]
                return result
    except json.JSONDecodeError:
        pass
    
    # If not JSON, use a more robust number extraction method
    # Compatible with "JSON-like" formats with text headers, e.g.: Reconstructed Matrices:{ [ ... ] }
    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]

    matrix = []
    for line in lines:
        numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
        if len(numbers) == 10:  # Treat 10 numbers per line as a valid matrix row
            matrix.append([int(round(float(x))) for x in numbers])

    # If extracting row by row fails, try to extract from the whole text and reshape to 10 columns
    if not matrix:
        all_numbers = re.findall(r'-?\d+(?:\.\d+)?', content)
        if len(all_numbers) >= 10 and len(all_numbers) % 10 == 0:
            arr = np.array([int(round(float(x))) for x in all_numbers], dtype=int)
            result = arr.reshape(-1, 10)
        else:
            result = np.array(matrix, dtype=int)
    else:
        result = np.array(matrix, dtype=int)
    
    # If the number of matrix rows exceeds 64, only take the first 64 rows
    if result.shape[0] > 64:
        print(f"Warning: File {file_path} has {result.shape[0]} rows, only using the first 64 rows.")
        result = result[:64, :]
    
    return result

def calculate_angular_difference(pred, true, mod):
    """
    Calculate angular difference considering wraparound.
    pred: predicted value
    true: ground truth value
    mod: modulo (64 for phi, 16 for psi)
    """
    # Calculate direct difference
    diff = np.abs(pred - true)
    
    # Consider wraparound: if difference is greater than mod/2, use wraparound distance
    wraparound_diff = mod - diff
    
    # Choose the smaller distance
    min_diff = np.minimum(diff, wraparound_diff)
    
    return min_diff

def calculate_nmse_with_wraparound(prediction, ground_truth):
    """
    Calculate NMSE considering quantization wraparound.
    phi-columns = indices {0,1,2,6,7} mod 64
    psi-columns = indices {3,4,5,8,9} mod 16
    """
    phi_indices = [0, 1, 2, 6, 7]  # phi column indices
    psi_indices = [3, 4, 5, 8, 9]  # psi column indices
    
    # Process phi and psi columns respectively
    total_squared_error = 0
    total_squared_true = 0
    
    # Process phi columns (mod 64)
    for idx in phi_indices:
        pred_col = prediction[:, idx]
        true_col = ground_truth[:, idx]
        
        # Calculate error considering wraparound
        errors = calculate_angular_difference(pred_col, true_col, 64)
        
        total_squared_error += np.sum(errors ** 2)
        total_squared_true += np.sum(true_col ** 2)
    
    # Process psi columns (mod 16)
    for idx in psi_indices:
        pred_col = prediction[:, idx]
        true_col = ground_truth[:, idx]
        
        # Calculate error considering wraparound
        errors = calculate_angular_difference(pred_col, true_col, 16)
        
        total_squared_error += np.sum(errors ** 2)
        total_squared_true += np.sum(true_col ** 2)
    
    # Calculate NMSE
    nmse = total_squared_error / total_squared_true
    
    return nmse

def calculate_detailed_errors(prediction, ground_truth):
    """
    Calculate detailed error statistics.
    """
    phi_indices = [0, 1, 2, 6, 7]
    psi_indices = [3, 4, 5, 8, 9]
    
    phi_errors = []
    psi_errors = []
    
    # Collect errors for phi columns
    for idx in phi_indices:
        errors = calculate_angular_difference(prediction[:, idx], ground_truth[:, idx], 64)
        phi_errors.extend(errors)
    
    # Collect errors for psi columns
    for idx in psi_indices:
        errors = calculate_angular_difference(prediction[:, idx], ground_truth[:, idx], 16)
        psi_errors.extend(errors)
    
    phi_errors = np.array(phi_errors)
    psi_errors = np.array(psi_errors)
    
    return phi_errors, psi_errors

def main():
    # Find the ground truth file (Truth.txt) in the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    gt_file = 'Truth.txt'
    if not os.path.exists(gt_file):
        print("Error: Ground truth file 'Truth.txt' not found in the current directory!")
        sys.exit(1)

    gt_name = os.path.basename(gt_file)
    recon_file = 'reconstructed_matrix.json'

    if not os.path.exists(recon_file):
        print(f"Error: Reconstructed matrix file {recon_file} not found.")
        sys.exit(1)

    print("Loading data...")
    print(f"Ground truth file: {gt_name}")
    ground_truth = load_matrix_from_file(gt_file)
    reconstructed = load_matrix_from_file(recon_file)

    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Reconstructed matrix shape: {reconstructed.shape}")

    if ground_truth.shape != reconstructed.shape:
        print(f"\nError: Matrix shape mismatch! Ground truth {ground_truth.shape} vs Reconstructed {reconstructed.shape}")
        sys.exit(1)

    print("\n" + "="*60)
    print("Calculate NMSE (considering quantization wraparound)")
    print("phi-columns (mod 64): indices {0,1,2,6,7}")
    print("psi-columns (mod 16): indices {3,4,5,8,9}")
    print("="*60)

    # Calculate NMSE
    nmse = calculate_nmse_with_wraparound(reconstructed, ground_truth)
    nmse_db = 10 * np.log10(nmse) if nmse > 0 else float('-inf')
    print(f"\nNMSE: {nmse:.6f} ({nmse_db:.2f} dB)")

    # Detailed error analysis
    print("\n" + "="*60)
    print("Detailed Error Analysis")
    print("="*60)
    phi_errors, psi_errors = calculate_detailed_errors(reconstructed, ground_truth)
    print(f"phi column errors - Mean: {phi_errors.mean():.3f}, Max: {phi_errors.max()}, Std Dev: {phi_errors.std():.3f}")
    print(f"psi column errors - Mean: {psi_errors.mean():.3f}, Max: {psi_errors.max()}, Std Dev: {psi_errors.std():.3f}")

    # Save results
    result_file = 'nmse_results.txt'
    print(f"\nSaving results to {result_file}")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("NMSE Calculation Results (considering quantization wraparound)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Ground truth: {gt_name}\n")
        f.write(f"Reconstructed matrix: {recon_file}\n\n")
        f.write("Configuration:\n")
        f.write("phi-columns (mod 64): indices {0,1,2,6,7}\n")
        f.write("psi-columns (mod 16): indices {3,4,5,8,9}\n\n")
        f.write(f"NMSE: {nmse:.6f} ({nmse_db:.2f} dB)\n\n")
        f.write("Detailed Error Analysis:\n")
        f.write(f"  phi column errors - Mean: {phi_errors.mean():.3f}, Max: {phi_errors.max()}, Std Dev: {phi_errors.std():.3f}\n")
        f.write(f"  psi column errors - Mean: {psi_errors.mean():.3f}, Max: {psi_errors.max()}, Std Dev: {psi_errors.std():.3f}\n")

    print("Results saved successfully!")

    # Convert reconstructed_matrix.json to txt format similar to ground truth
    recon_txt_file = 'reconstructed_matrix.txt'
    reconstructed_int = np.rint(reconstructed).astype(int)
    recon_str = np.array2string(reconstructed_int, separator=' ', threshold=np.inf, max_line_width=np.inf)
    with open(recon_txt_file, 'w', encoding='utf-8') as f:
        f.write(recon_str)
    print(f"Reconstructed matrix converted and saved as: {recon_txt_file}")

if __name__ == "__main__":
    main()
