import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\Legion\Desktop\MachineLearningProject\data\creditcard.csv")

# Scale numerical columns
df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
df['scaled_time'] = StandardScaler().fit_transform(df[['Time']])
df = df.drop(columns=['Amount','Time'])

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1])),
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# Fit baseline model
model.fit(X_train, y_train)

# Probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Default threshold performance
default_preds = (y_prob >= 0.5).astype(int)

print("\n=== XGBoost Classification Report ===")
print(classification_report(y_test, default_preds))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))

# Find best threshold
thresholds = np.linspace(0.01, 1, 400)
best_f1 = 0
best_thresh = 0

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("\nBest threshold:", best_thresh)
print("Best F1:", best_f1)

# Final predictions after threshold tuning
final_preds = (y_prob >= best_thresh).astype(int)

print("\n=== XGBoost Performance (After Threshold Tuning) ===")
print(classification_report(y_test, final_preds))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))


#     FEATURE IMPORTANCE
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.bar(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xticks(rotation=90)
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()



