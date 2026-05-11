import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

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

# Baseline Random Forest
rf_baseline = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_baseline.fit(X_train, y_train)

y_prob = rf_baseline.predict_proba(X_test)[:, 1]
threshold = 0.27
final_preds = (y_prob >= threshold).astype(int)

print("\nRandom Forest Classification Report (Threshold = 0.27, All Features):")
print(classification_report(y_test, final_preds))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))
print("F1 Score:", f1_score(y_test, final_preds))

# Feature importances
importances = rf_baseline.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sorted_idx = feature_importance_df['importance'].sort_values(ascending=False).index
plt.bar(range(len(sorted_idx)), feature_importance_df['importance'][sorted_idx])
plt.xticks(range(len(sorted_idx)), feature_importance_df['feature'][sorted_idx], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()


# Evaluate top N features
for N in [20, 15, 10, 5]:
    top_features = feature_importance_df['feature'][:N].values
    X_train_sel = X_train[top_features]
    X_test_sel = X_test[top_features]

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_sel, y_train)

    y_prob_sel = rf.predict_proba(X_test_sel)[:, 1]
    final_preds_sel = (y_prob_sel >= threshold).astype(int)

    print(f"\nRandom Forest Classification Report (Threshold = 0.27, Top {N} Features):")
    print(classification_report(y_test, final_preds_sel))
    print(f"F1 Score (Top {N} features):", f1_score(y_test, final_preds_sel))
