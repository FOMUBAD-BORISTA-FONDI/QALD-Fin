# Combining Divide and Conquer Rule with Large Language Models for Question Answering

## Overview

This project explores the integration of the Divide and Conquer algorithm with Large Language Models (LLMs) to enhance question answering capabilities. Our methodology was evaluated in the Schorlaly Hybrid QALD Challenge hosted by the ISWC conference. The challenge involved predicting answers to questions from various domains using datasets provided for both training and testing.

## Key Components

1. **Divide and Conquer Algorithm**: This approach segments datasets into manageable parts, enabling focused analysis and improved prediction accuracy.

2. **Large Language Models (LLMs)**: We utilized BERT (Bidirectional Encoder Representations from Transformers) for understanding and generating responses.

3. **Hybrid Methodology**: Our method combines LLMs with traditional rule-based approaches to optimize question answering performance.

## Results

- **F1-Score Improvement**: The initial dataset, focusing on authors, achieved an F1-score of 0.1964. Applying BERT combined with the Divide and Conquer algorithm improved the F1-score to 0.285, earning us second place for this dataset.

## Methodology

The approach involved several key steps:
1. **Data Conversion and Re-organization**: Transforming and re-organizing datasets for efficient question analysis.
2. **Data Segmentation**: Dividing datasets based on categories such as authors and institutions.
3. **Keyword Extraction**: Identifying key terms to guide the segmentation and categorization.
4. **Program Development and Prediction**: Creating a custom program to retrieve and predict answers, integrating LLMs when necessary.

For a detailed description, refer to the [Paper](https://github.com/FOMUBAD-BORISTA-FONDI/QALD-Fin/tree/master/paper).

## How to run the project

## Experimentation Environment

- **Dataset Overview**: 
    - Training Size: 5000
    - Test Size: 702

- **Hardware and Software**:
    - **CPU**: Intel Core i7-6820HQ, 2.70GHz, 8 cores
    - **RAM**: 24.0 GiB
    - **Disk**: 512.2 GB
    - **OS**: Ubuntu 24.04.4 LTS

## Important Notes

- **Answer Variability**: Predictions made by LLMs may vary. The results are influenced by the model's training data and inherent randomness in language generation. Therefore, answers provided by the LLM might not be consistent across different runs.

## References

- Manning, C. D., Raghavan, P., & Schütze, H. (2014). *Introduction to Information Retrieval*. MIT Press.
- Jurafsky, D., & Martin, J. H. (2020). *Speech and Language Processing* (3rd ed.). Pearson.
- Lin, J., Zhang, Y., Liu, Z., & Sun, H. (2021). “A hybrid approach to question answering with knowledge graphs and neural networks.” *Journal of Artificial Intelligence Research*, 70, 1001-1030.
- Wang, F., Zhang, X., & Gao, J. (2022). “Combining retrieval-based and generative models for hybrid question answering.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(1), 1234-1242.
- Brown, T. B., Mann, B., Ryder, N., et al. (2020). “Language models are few-shot learners.” *arXiv preprint arXiv:2005.14165*.
- Radford, A., Johnson, C., & Sutskever, I. (2021). “Learning transferable visual models from natural language supervision.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 8748-8760.
- Liu, H., Zhang, R., & Zhang, M. (2023). “HybridQA: A hybrid approach for question answering.” *IEEE Transactions on Knowledge and Data Engineering*, 35(2), 556-568.

## License
