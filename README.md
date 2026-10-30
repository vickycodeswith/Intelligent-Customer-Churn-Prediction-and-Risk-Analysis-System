# Intelligent Customer Churn Prediction and Risk Analysis System

An end-to-end Machine Learning application that predicts customer churn and identifies customers who may be at risk of leaving a telecom service provider. The system uses customer demographic, service, contract, and billing information to generate churn predictions, probability scores, and risk categories through an interactive Streamlit web application.

## 🚀 Live Demo

Experience the **Intelligent Customer Churn Prediction and Risk Analysis System** live:

🔗 **[Launch Live Application](https://intelligent-customer-churn-prediction-and-risk-analysis-system.streamlit.app/)**

The application allows users to enter customer details and instantly receive:
- Customer churn predictions
- Churn probability scores
- Customer risk classifications (Low, Medium, High)

**Try the live application and explore how Machine Learning can help identify customers at risk of leaving.**
## Project Overview

Customer churn is a major challenge for telecom companies because losing existing customers can affect revenue and business growth.

This project develops a machine learning solution to identify customers who are likely to discontinue their services. It uses historical customer data to train a classification model and provides an interactive interface for generating predictions on new customer records.

## Key Features

- **Customer Churn Prediction:** Predicts whether a customer is likely to leave.
- **Churn Probability:** Provides the estimated probability of customer churn.
- **Risk Classification:** Categorizes customers into Low, Medium, and High Risk.
- **Data Preprocessing:** Handles missing values, categorical features, and numerical features.
- **Class Imbalance Handling:** Uses SMOTE to address the imbalance between churn and non-churn customers.
- **Machine Learning Pipeline:** Integrates preprocessing and prediction into one reusable pipeline.
- **Interactive Web Application:** Provides a user-friendly interface using Streamlit.
- **Model Persistence:** Saves and loads the trained model using Joblib.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Imbalanced Data | Imbalanced-learn (SMOTE) |
| Model Deployment | Streamlit |
| Model Serialization | Joblib |
| Development Environment | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains customer information such as:

- Customer demographics
- Account information
- Internet and phone services
- Contract details
- Monthly and total charges
- Customer churn status

**Target Variable:** `Churn`

- `Yes` – Customer has left the company.
- `No` – Customer has stayed with the company.

The dataset is used for data preprocessing, exploratory data analysis, model training, and evaluation.

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering and Encoding
5. Train, Validation, and Test Split
6. Baseline Model Training
7. Class Imbalance Handling using SMOTE
8. Model Selection and Validation
9. Final Model Training
10. Model Evaluation
11. Model Serialization
12. Streamlit Application Development

## Model Performance

The selected model is a **Logistic Regression classifier with SMOTE**, integrated into a machine learning pipeline.

The following results were obtained on the held-out test dataset:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 74.93% |
| Precision | 51.91% |
| Recall | 77.22% |
| F1 Score | 62.09% |
| ROC-AUC | 84.41% |

### Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 575 | 201 |
| Churn | 64 | 217 |

**Model Insight:** The model achieves 77.22% recall, helping identify a substantial proportion of customers who actually churn. However, its precision indicates that some customers predicted to churn may not actually leave.

## Application Output

The Streamlit application accepts customer information and displays:

- Churn Prediction
- Churn Probability
- Risk Category

Risk categories are based on the following initial probability thresholds:

| Churn Probability | Risk Category |
|---|---|
| Below 30% | Low Risk |
| 30% to below 60% | Medium Risk |
| 60% or above | High Risk |

The churn prediction itself uses the classifier's default decision threshold.

## Project Structure

```text
Intelligent-Customer-Churn-Prediction-and-Risk-Analysis-System/
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── customer_churn_pipeline.pkl
│
├── churn_prediction.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vickycodeswith/Intelligent-Customer-Churn-Prediction-and-Risk-Analysis-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Intelligent-Customer-Churn-Prediction-and-Risk-Analysis-System
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Model File

The trained machine learning pipeline is stored at:

```text
models/customer_churn_pipeline.pkl
```

The application loads this saved pipeline to make predictions without retraining the model.

## Future Improvements

- Improve model precision while maintaining strong churn recall.
- Add customer-level explanations for predictions.
- Develop a customer churn analytics dashboard.
- Evaluate model performance on new customer data.
- Deploy the application to a cloud platform.

## Limitations

- Predictions depend on the quality and characteristics of the training dataset.
- The model's performance may change when applied to different customer populations.
- Risk categories are initial business thresholds and are not calibrated risk estimates.
- Predictions support business decisions but should not be treated as guaranteed outcomes.

## Author

**Vicky Yadav**

Computer Engineering Student | Machine Learning & AI Enthusiast

GitHub: [@vickycodeswith](https://github.com/vickycodeswith)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
