# Project Statement

## Problem Statement
Credit card fraud is a major concern for financial institutions. With the increasing volume of online transactions, the challenge lies in distinguishing fraudulent transactions from legitimate ones in real-time. The dataset for this problem is highly imbalanced, with fraudulent transactions representing a very small fraction of the total. This imbalance makes it difficult for standard classification models to learn the patterns of fraud, often leading to a high number of false negatives (i.e., failing to detect actual fraud).

The goal is to build a machine learning model that can accurately detect fraudulent credit card transactions while minimizing false negatives, thereby ensuring high recall or sensitivity.

## Scope of the project
The scope of this project includes:
- **Data Preprocessing**: Normalizing features (Amount, Time) using RobustScaler to handle outliers.
- **Handling Imbalance**: Implementing the Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic examples for the minority class (fraud) in the training data.
- **Model Development**: Training a RandomForestClassifier on the balanced dataset to learn complex patterns.
- **Evaluation**: Testing the model on a realistic, imbalanced test set to ensure real-world applicability, focusing on Recall as a key metric.
- **Deployment/Demo**: Developing a simple web-based demonstration using Streamlit to interact with the model.

## Target users
- **Financial Institutions**: Banks and credit card companies looking to reduce financial losses due to fraud.
- **Cybersecurity Analysts**: Professionals responsible for monitoring and securing transaction systems.
- **Data Scientists/ML Engineers**: Practitioners interested in techniques for handling imbalanced datasets in fraud detection.

## High-level features
- **Real-time Fraud Detection**: Capability to classify transactions as fraudulent or legitimate.
- **Imbalanced Data Handling**: robust training pipeline utilizing SMOTE to improve detection of rare fraud events.
- **Interactive Web Interface**: A Streamlit-based UI for users to input transaction details and get predictions.
- **Performance Metrics**: Detailed evaluation reports including Precision, Recall, and F1-score to assess model effectiveness.
