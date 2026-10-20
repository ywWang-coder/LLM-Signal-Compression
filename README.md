# Introduction
This repository contains the dataset, prompt details and experimental results for the paper:  "Can LLMs Compress Wireless Signals? Physics-Informed Prompting for Training-Free Signal Compression." This paper has been accepted to **Findings of EMNLP 2026**.

## Paper

The full paper is available in this repository: [Read the paper (PDF)](EMNLP_26.pdf).

## Authors

- **Yiwei Wang**<sup>1</sup> - wangyw12024@shanghaitech.edu.cn
- **Lixiang Lian**<sup>1,*</sup> - lianlx@shanghaitech.edu.cn
- **Sijie Ji**<sup>2,*</sup> - sijieji@caltech.edu

<sup>1</sup> School of Information Science and Technology, ShanghaiTech University, China  
<sup>2</sup> Division of Engineering and Applied Science, California Institute of Technology, USA  
<sup>*</sup> Corresponding authors

## Physics-Informed Prompt
This section contains the core experiments of our proposed physics-informed prompt framework. It provides detailed experimental results, evaluation scripts, and LLM logs under different Prompt Configurations that we designed. We also provide the results of our prompts on open-source LLMs. For more details, please navigate to the `Physicis_informed Prompt/` directory and carefully read the `README.md` file in that folder.

## Evaluation
Due to the extensive volume of experimental results for LLMs, it is impractical to display them in their entirety. Therefore, we show one representative sequence for each baseline method and each LLM experimental settings for comparison. The corresponding test data can be found in the `data/test/test_data` while the LLM experimental results are presented via links. We provide the complete evaluation dataset in the `data/test/more_test_data` directory. This includes 20 sequences per scenario, totaling 200 wireless signal frames available for comprehensive performance assessment. 

## License

This repository is released under the [MIT License](LICENSE). You may use,
copy, modify, and distribute the repository's contents, provided that the
copyright notice and license text are retained.

Third-party services, linked content, model outputs, and any third-party
materials remain subject to their respective terms and licenses.
