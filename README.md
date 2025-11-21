# FraudGuard: Real-Time Financial Fraud Detection

## Domain
Financial Services & Cybersecurity

## Problem Statement
Credit card fraud is a major concern for financial institutions. With the increasing volume of online transactions, the challenge lies in distinguishing fraudulent transactions from legitimate ones in real-time. The dataset for this problem is highly imbalanced, with fraudulent transactions representing a very small fraction of the total. This imbalance makes it difficult for standard classification models to learn the patterns of fraud, often leading to a high number of false negatives (i.e., failing to detect actual fraud).

The goal is to build a machine learning model that can accurately detect fraudulent credit card transactions while minimizing false negatives, thereby ensuring high recall or sensitivity.

## Solution Approach
To address the class imbalance problem, we will use the **Synthetic Minority Over-sampling Technique (SMOTE)**.

### What is SMOTE?
SMOTE is a powerful technique to handle imbalanced datasets. Instead of simply duplicating existing minority class instances, SMOTE creates new, synthetic instances. It works by:
1.  **Identifying** the minority class instances (fraudulent transactions in our case).
2.  **Finding** the k-nearest neighbors for each minority instance.
3.  **Creating** synthetic instances along the line segments joining the minority instance and its selected neighbors.

By generating synthetic examples, SMOTE helps the model learn the characteristics of the minority class more effectively, leading to better classification performance and, crucially, higher recall.

Our approach is as follows:
1.  **Data Preprocessing**: We will normalize the 'Amount' and 'Time' features using `RobustScaler`, which is less sensitive to outliers.
2.  **Train-Test Split**: The data will be split into training and testing sets.
3.  **SMOTE Application**: We will apply SMOTE **only to the training set** to prevent data leakage and ensure our model is evaluated on a realistic, imbalanced test set.
4.  **Model Training**: A `RandomForestClassifier` will be trained on the balanced training data. Random Forest is an ensemble method that is robust and performs well in complex classification tasks.
5.  **Evaluation**: The model will be evaluated on the original, imbalanced test set.

## Technologies Used
- **Python**: The core programming language.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning tasks (RobustScaler, RandomForestClassifier, train_test_split, classification_report).
- **Imbalanced-learn**: For the SMOTE implementation.
- **Streamlit**: To create a simple web-based demo of the model.
- **Joblib**: For saving and loading the trained model and scaler.
- **Matplotlib**: For generating plots and visualizations.

## How to Run
1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your-repo-url>
    cd FraudGuard
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Train the model:**
    - Place the `creditcard.csv` dataset in a `dataset/` directory.
    - Run the training script:
    ```bash
    python src/train_model.py
    ```
    This will save the trained model (`fraud_model.joblib`) and the scaler (`scaler.joblib`) into a `models/` directory.

4.  **Run the Streamlit application:**
    ```bash
    streamlit run src/app.py
    ```
    Open your browser and navigate to the provided local URL to interact with the demo.

## Results
The model's performance is evaluated based on its ability to correctly identify fraudulent transactions. In fraud detection, **Recall (or Sensitivity)** is the most important metric. A high recall means the model is effective at catching a high percentage of actual fraud cases.

After training, the model achieved the following performance on the imbalanced test set:

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.85      0.83      0.84       148

    accuracy                           1.00     85443
   macro avg       0.92      0.91      0.92     85443
weighted avg       1.00      1.00      1.00     85443
```

The high recall for the minority class (fraudulent transactions) indicates that our approach using SMOTE and a RandomForestClassifier was successful in building a model that can effectively identify fraud.
