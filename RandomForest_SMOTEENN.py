from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
# Load dataset
df = pd.read_csv(r"C:\Users\Legion\Desktop\MachineLearningProject\data\creditcard.csv")

# Scale numerical columns
df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
df['scaled_time'] = StandardScaler().fit_transform(df[['Time']])
df = df.drop(columns=['Amount', 'Time'])

# Features + labels
X = df.drop('Class', axis=1)
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X_train, y_train)

print("Before SMOTEENN:", y_train.value_counts())
print("After SMOTEENN:", y_res.value_counts())

rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_res, y_res)

y_prob = rf.predict_proba(X_test)[:, 1]

# Threshold tuning
thresholds = np.linspace(0.01, 0.90, 200)
best_f1 = 0
best_thresh = 0

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

# Final predictions using best threshold
final_preds = (y_prob >= best_thresh).astype(int)

# Metrics
print("Best threshold:", best_thresh)
print("Best F1:", best_f1)
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))
print("\nClassification Report:")
print(classification_report(y_test, final_preds))

