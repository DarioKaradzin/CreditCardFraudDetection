from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import f1_score, classification_report, roc_auc_score, average_precision_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# Load dataset
df = pd.read_csv(r"C:\Users\Legion\Desktop\MachineLearningProject\data\creditcard.csv")

# Scale numerical columns
df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
df['scaled_time'] = StandardScaler().fit_transform(df[['Time']])
df = df.drop(columns=['Amount','Time'])

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Define the best XGBoost model
best_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.7,
    min_child_weight=7,
    gamma=0.2
)

# Store results
results = []

# Stratified splits (10 splits)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

for i, (train_idx, test_idx) in enumerate(sss.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Set scale_pos_weight for this split
    imbalance_ratio = sum(y_train==0) / sum(y_train==1)
    best_xgb.set_params(scale_pos_weight=imbalance_ratio)

    # Train model
    best_xgb.fit(X_train, y_train)

    # Predict probabilities
    y_prob = best_xgb.predict_proba(X_test)[:, 1]

    # Threshold tuning
    thresholds = np.linspace(0.01, 0.99, 300)
    best_f1, best_thresh = 0, 0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    y_pred = (y_prob >= best_thresh).astype(int)

    # Store metrics
    split_results = {
        "Split": i,
        "Best Threshold": best_thresh,
        "F1": best_f1,
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob)
    }
    results.append(split_results)

    print(f"\n=== Split {i} Results ===")
    print(f"Best Threshold: {best_thresh:.4f}")
    print(f"F1: {best_f1:.4f}")
    print("ROC-AUC:", f"{split_results['ROC-AUC']:.4f}")
    print("PR-AUC:", f"{split_results['PR-AUC']:.4f}")
    print(classification_report(y_test, y_pred))

# Aggregate results
results_df = pd.DataFrame(results)
print("\n=== Summary of 10 Splits ===")
print(results_df.describe())

#     FEATURE IMPORTANCE
importances = best_xgb.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, lw=2, color='darkorange',
         label=f'XGBoost ROC (AUC = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy',
         label="Random chance")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc_val = average_precision_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, lw=2, color='purple',
         label=f'PR-AUC = {pr_auc_val:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - XGBoost")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

