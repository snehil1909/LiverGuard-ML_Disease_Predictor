# LiverGuard: Disease Predictor

## Overview
Liver diseases are a significant global health issue, leading to high mortality rates and immense pressure on healthcare systems. **LiverGuard** aims to mitigate this by leveraging machine learning (ML) techniques for early detection and diagnosis of liver diseases. The project focuses on accurate predictions to aid medical professionals in clinical decision-making.

## Features
- Preprocessing techniques including handling null values, label encoding, and outlier detection.
- Advanced feature engineering for improved model performance.
- Evaluation of multiple ML models with metrics such as accuracy and F1-score.
- Implementation of ensemble methods like Random Forest and XGBoost for robust predictions.

## Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset/data)
- **Description**:
  - **Rows**: 30,691
  - **Features**: 11, including patient demographics, liver function markers, and protein levels.
  - **Target Variable**: Binary classification (1: Liver Disease, 0: No Liver Disease)

## Data Processing
### Steps
1. **Null Value Handling**:
   - Numerical columns: Filled with mean values.
   - Categorical columns: Filled with mode.
2. **Label Encoding**:
   - Gender: Male (0), Female (1).
   - Labels: 1 (Liver Disease), 2 (Non-Liver Disease) mapped to binary values (1, 0).
3. **Splitting**:
   - Train:Val:Test = 70:10:20.
4. **Outlier Handling**:
   - Z-Score method to remove extreme values.

## Methodology
### Feature Engineering
- Polynomial features for Total and Direct Bilirubin.
- New feature: Albumin-to-Protein Ratio.
- Log transformations for skewed data.
- Recursive Feature Elimination (RFE) for top 6 features.

### Model Evaluation
- Baseline Models: Naive Bayes, Logistic Regression.
- Ensemble Methods: Random Forest, XGBoost.
- Advanced Models: Support Vector Machine (SVM), Multi-Layer Perceptrons (MLP).
- Evaluation Metrics: Accuracy, F1-score, RMSE, Classification Reports.

## Results
- **Best Performing Models**:
  - Random Forest: **99.68% accuracy**.
  - XGBoost: **99.79% accuracy**.
- Simpler models like Naive Bayes and Logistic Regression underperformed due to dataset complexity.

## Challenges
- Imbalanced dataset addressed through bootstrapping.
- Computational cost of ensemble methods.
- Dataset representativeness may limit real-world generalization.

## Contribution
- **Snehil Jaiswal**: Dataset analysis, preprocessing, literature review, report writing.
- **Sneha Nagpal**: Model evaluation, EDA, feature extraction.
- **Sidak Singh Chahal**: Literature review, result inference, presentation.
- **Shubham Kumar Dwivedi**: EDA, graph plotting, presentation.

## References
1. Tokala, Srilatha et al. *Liver Disease Prediction and Classification using Machine Learning Techniques*, 2023.
2. Bhupathi, Deepika et al. *Liver disease detection using machine learning techniques*, 2022.
3. Kaggle Dataset: [Liver Disease Patient Dataset](https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset/data)
4. Additional references from [IEEE](https://ieeexplore.ieee.org) and [ScienceDirect](https://www.sciencedirect.com).

## Repository
Find the complete code and documentation [here](https://github.com/snehil1909/ML_Project.git).
