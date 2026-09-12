# Ablation-Driven Optimization of Spiking Neural Networks for Bank Account Fraud Detection

Source code of the paper entitled "Ablation-Driven Optimization of Spiking Neural Networks for Bank Account Fraud Detection" accepted at "IDEAL 2026", the 27th International Conference on Intelligent Data Engineering and Automated Learning.

## Abstract
Spiking neural networks (SNNs) have emerged as the next paradigm in artificial intelligence (AI), offering potential energy efficiency for real-world deployments. However, training these networks on highly imbalanced datasets remains challenging due to their high-dimensional parameter spaces. This paper presents three ablation studies on optimization strategies for SNNs in classification tasks, analyzing membrane decay, firing thresholds, temporal window size, and surrogate gradient slopes. Our results indicate that using fixed parameter initialization while learning only the threshold, combined with smaller temporal windows and architecture-specific surrogate gradient slopes, yields the most stable performance. We evaluate these optimized SNNs on the Bank Account Fraud suite against traditional machine learning algorithms. While boosting ensembles maintain strong performance for this classification task, our optimized SNNs demonstrate competitive performance measured by recall under a 5% false positive rate, while achieving higher fairness in terms of predictive equality of sensitive attributes.

**Keywords:** Spiking Neural Networks $\cdot$ Neuromorphic Computing $\cdot$ Neuronal Dynamics $\cdot$ Bayesian Optimization $\cdot$ Responsible AI.

## Installation

To install the required packages, run the following command:
```sh
pip install -r requirements.txt
```
Download the six Variant of the Bank Account Fraud (BAF) Dataset and extract the parquet files to the data folder.

## Dataset

The Bank Account Fraud (BAF) dataset is a synthetic dataset based on real-world data that simulates bank account opening applications. The dataset contains 6 parquet files, each representing a different variant of the dataset (Base, Variant I, Variant II, Variant III, Variant IV, and Variant V). It contains 30 features and a binary target variable indicating whether the application is fraudulent or not.

## Repository Structure

The repository is structured as follows:

- `data`: Contains the Bank Account Fraud dataset.
- `src`: Contains the source code of the project.


## Bibtex

To cite this work, use the following bibtex entry:
```bibtex
TBD
```
## Issues

This code is imported and adapted from the original research repository. Consequently, the code may contain bugs or issues. If you encounter any issues while running the code, please open an issue in the repository.
