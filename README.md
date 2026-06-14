# Customer Churn Prediction

## Project Overview

This project predicts whether a telecom customer is likely to leave (churn) a service using Machine Learning. It includes data preprocessing, exploratory data analysis (EDA), data visualization, model training, and performance evaluation using a Random Forest Classifier.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset

- **Dataset:** Telco Customer Churn Dataset
- **Total Records:** 7,043
- **Features:** 21

---

## Machine Learning Model

- Random Forest Classifier

---

## Results

| Metric | Value |
|---------|-------|
| Accuracy | **78.92%** |
| ROC-AUC Score | **0.6834** |

---

## Project Visualizations

### 1. Customer Churn Distribution

![Customer Churn Distribution](images/01_churn_distribution.png)

---

### 2. Monthly Charges vs Churn

![Monthly Charges vs Churn](images/02_monthly_charges_vs_churn.png)

---

### 3. Contract Type vs Churn

![Contract Type vs Churn](images/03_contract_vs_churn.png)

---

### 4. Confusion Matrix

![Confusion Matrix](images/04_confusion_matrix.png)

---

### 5. Feature Importance

![Feature Importance](images/05_feature_importance.png)

---

## Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   └── telco_churn.csv
│
├── images/
│   ├── 01_churn_distribution.png
│   ├── 02_monthly_charges_vs_churn.png
│   ├── 03_contract_vs_churn.png
│   ├── 04_confusion_matrix.png
│   └── 05_feature_importance.png
│
├── customer.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/durgamsneha99/Customer-_Churn_Prediction.git
```

### 2. Open the project

```bash
cd Customer-_Churn_Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python customer.py
```

---

## Future Improvements

- Improve model accuracy using XGBoost or LightGBM.
- Perform hyperparameter tuning.
- Build a web application using Flask or Streamlit.
- Deploy the model for real-time predictions.

---

## Author

**Sneha Durgam**

GitHub: https://github.com/durgamsneha99
