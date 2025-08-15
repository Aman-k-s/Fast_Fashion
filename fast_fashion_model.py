# fast_fashion_logreg.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_excel("HM-Sales-2018.xlsx")

# Keep only relevant columns
df = df[["Category", "Sub-Category", "Sales", "Quantity", "Discount", "Profit"]]

# Create Target: 1 = High Risk (unsustainable), 0 = Low Risk
# Example rule: High discount or negative profit = high risk
df["Risk"] = df.apply(lambda x: 1 if (x["Discount"] > 0.5 or x["Profit"] < 0) else 0, axis=1)

# Encode categorical columns
le_cat = LabelEncoder()
le_subcat = LabelEncoder()
df["Category"] = le_cat.fit_transform(df["Category"])
df["Sub-Category"] = le_subcat.fit_transform(df["Sub-Category"])

# Features & Target
X = df[["Category", "Sub-Category", "Sales", "Quantity", "Discount", "Profit"]]
y = df["Risk"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numeric data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save Model & Preprocessors
joblib.dump(log_reg, "logreg_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_cat, "labelencoder_category.pkl")
joblib.dump(le_subcat, "labelencoder_subcategory.pkl")
