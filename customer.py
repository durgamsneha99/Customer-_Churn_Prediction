import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/telco_churn.csv")

# ===============================
# Display First 5 Rows
# ===============================
print("First 5 Rows:")
print(df.head())

# ===============================
# Dataset Information
# ===============================
print("\nDataset Information:")
print(df.info())

# ===============================
# Missing Values
# ===============================
print("\nMissing Values:")
print(df.isnull().sum())

# ===============================
# Statistical Summary
# ===============================
print("\nStatistical Summary:")
print(df.describe())

# ===============================
# Graph 1: Customer Churn Distribution
# ===============================
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("images/churn_distribution.png")

plt.show()

# ===============================
# Graph 2: Monthly Charges vs Churn
# ===============================
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("images/churn_distribution.png")

plt.show()

# ===============================
# Graph 3: Contract Type vs Churn
# ===============================
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/churn_distribution.png")

plt.show()

# ===============================
# Data Preprocessing
# ===============================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Remove customerID
df = df.drop("customerID", axis=1)

# Convert Churn to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

print("\nData after preprocessing:")
print(df.head())

# ===============================
# Split Features and Target
# ===============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# Train Random Forest Model
# ===============================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================
y_pred = model.predict(X_test)

# ===============================
# Model Evaluation
# ===============================
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nROC-AUC Score:", roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion Matrix
# ===============================
plt.figure(figsize=(6,5))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("images/churn_distribution.png")

plt.show()

# ===============================
# Feature Importance
# ===============================
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features Affecting Customer Churn")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("images/churn_distribution.png")

plt.show()