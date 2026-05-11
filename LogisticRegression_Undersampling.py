import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_validate
from imblearn.under_sampling import RandomUnderSampler

# Load dataset
df = pd.read_csv(r"C:\Users\Legion\Desktop\MachineLearningProject\data\creditcard.csv")

# Scale numerical columns
df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
df['scaled_time'] = StandardScaler().fit_transform(df[['Time']])

df = df.drop(columns=['Amount', 'Time'])

# Split features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. Undersample the majority class
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

print("Before undersampling:", y_train.value_counts())
print("After undersampling:", y_res.value_counts())

# 2. Train logistic regression on undersampled data
log_reg_rus = LogisticRegression(max_iter=500, solver='lbfgs')
log_reg_rus.fit(X_res, y_res)

# 3. Predict on TEST set (NOT undersampled!)
y_pred_rus = log_reg_rus.predict(X_test)
y_prob_rus = log_reg_rus.predict_proba(X_test)[:, 1]

# 4. Evaluate
print("\nClassification Report (Undersampling):")
print(classification_report(y_test, y_pred_rus))

print("ROC-AUC:", roc_auc_score(y_test, y_prob_rus))
print("PR-AUC:", average_precision_score(y_test, y_prob_rus))

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.arange(0.99, 1.001, 0.001)

print("Threshold | Precision | Recall | F1")
for t in thresholds:
    y_pred_t = (y_prob_rus >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    f = f1_score(y_test, y_pred_t)
    print(f"{t:.3f}       {p:.3f}      {r:.3f}    {f:.3f}")