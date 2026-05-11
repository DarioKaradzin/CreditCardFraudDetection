from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

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


# 1. Apply SMOTE on the training data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_smote.value_counts())

# 2. Train logistic regression using class weights
log_reg_smote_cw = LogisticRegression(
    max_iter=500,
    solver='lbfgs',
    class_weight='balanced'
)

log_reg_smote_cw.fit(X_smote, y_smote)

# 3. Predict on SMOTEd test set
y_pred = log_reg_smote_cw.predict(X_test)
y_prob = log_reg_smote_cw.predict_proba(X_test)[:, 1]

# 4. Evaluate
print("\nClassification Report (SMOTE + Class Weight):")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.arange(0.99, 1.001, 0.001)

print("Threshold | Precision | Recall | F1")
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    f = f1_score(y_test, y_pred_t)
    print(f"{t:.3f}       {p:.3f}      {r:.3f}    {f:.3f}")
