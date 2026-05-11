import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap


# Load + Preprocess
df = pd.read_csv(r"C:\Users\Legion\Desktop\MachineLearningProject\data\creditcard.csv")

df["scaled_amount"] = StandardScaler().fit_transform(df[["Amount"]])
df["scaled_time"] = StandardScaler().fit_transform(df[["Time"]])
df = df.drop(columns=["Amount", "Time"])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compute imbalance ratio for scale_pos_weight
pos = sum(y_train == 1)
neg = sum(y_train == 0)
imbalance_ratio = neg / pos

# XGBoost Hyperparameter Tuning
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1
)

param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.05, 0.1, 0.2],
    "scale_pos_weight": [imbalance_ratio, imbalance_ratio/2, imbalance_ratio*1.5]
}

search = RandomizedSearchCV(
    xgb,
    param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

best_xgb = search.best_estimator_
print("\nBest Hyperparameters:", search.best_params_)


# Evaluate Model
y_prob = best_xgb.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.01, 0.99, 300)
best_f1, best_thresh = 0, 0

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

y_pred = (y_prob >= best_thresh).astype(int)

print(f"\nBest threshold: {best_thresh}")
print(f"Best F1: {best_f1:.4f}\n")

print("=== XGBoost Performance ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))


# Feature Importance Plot (All Features)
importances = best_xgb.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(range(len(importances)), X.columns[sorted_idx], rotation=90)
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()
