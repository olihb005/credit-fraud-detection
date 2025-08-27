# =============================================================================
# Credit Card Fraud Detection - EDA with Model Comparison
# Author: Oliver HB
# Date: 28/8/25
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Create output directory
import os
if not os.path.exists('./output'):
    os.makedirs('./output')

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Load and explore data
print("="*60)
print("CREDIT CARD FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
print("="*60)

df = pd.read_csv('./data/creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 2. Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# 3. Class distribution
print("\nClass Distribution:")
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"Fraud percentage: {class_counts[1]/len(df)*100:.4f}%")

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum().max())

# 5. Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=df, palette=['green', 'red'])
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
plt.ylabel('Count')
plt.savefig('./output/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Amount distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df[df['Class']==0]['Amount'], bins=50, color='green')
plt.title('Legitimate Transaction Amounts')
plt.yscale('log')

plt.subplot(1, 2, 2)
sns.histplot(df[df['Class']==1]['Amount'], bins=50, color='red')
plt.title('Fraudulent Transaction Amounts')
plt.yscale('log')
plt.tight_layout()
plt.savefig('./output/amount_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Time distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df[df['Class']==0]['Time']/3600, bins=48, color='green')
plt.title('Legitimate Transaction Times')

plt.subplot(1, 2, 2)
sns.histplot(df[df['Class']==1]['Time']/3600, bins=48, color='red')
plt.title('Fraudulent Transaction Times')
plt.tight_layout()
plt.savefig('./output/time_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.savefig('./output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Model preparation
print("\n" + "="*60)
print("MODEL PREPARATION")
print("="*60)

X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training fraud cases: {y_train.sum()}")
print(f"Test fraud cases: {y_test.sum()}")

# Scale data for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Isolation Forest Model
print("\n" + "="*60)
print("ISOLATION FOREST MODEL")
print("="*60)

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.0017,  # Match actual fraud rate
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_train_scaled)
y_pred_iso = iso_forest.predict(X_test_scaled)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convert -1 to 1

# Evaluate Isolation Forest
print("Isolation Forest Results:")
print(classification_report(y_test, y_pred_iso, target_names=['Legitimate', 'Fraudulent']))

cm_iso = confusion_matrix(y_test, y_pred_iso)
print(f"False Positives: {cm_iso[0,1]}")
print(f"False Negatives: {cm_iso[1,0]}")

# 11. Random Forest Model
print("\n" + "="*60)
print("RANDOM FOREST MODEL")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# Evaluate Random Forest
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate', 'Fraudulent']))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"False Positives: {cm_rf[0,1]}")
print(f"False Negatives: {cm_rf[1,0]}")

# 12. Model Comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Create comparison dataframe
comparison = pd.DataFrame({
    'Metric': ['False Positives', 'False Negatives', 'Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)'],
    'Isolation Forest': [
        cm_iso[0,1],
        cm_iso[1,0],
        cm_iso[1,1]/(cm_iso[1,1]+cm_iso[0,1]),
        cm_iso[1,1]/(cm_iso[1,1]+cm_iso[1,0]),
        2*cm_iso[1,1]/(2*cm_iso[1,1]+cm_iso[0,1]+cm_iso[1,0])
    ],
    'Random Forest': [
        cm_rf[0,1],
        cm_rf[1,0],
        cm_rf[1,1]/(cm_rf[1,1]+cm_rf[0,1]),
        cm_rf[1,1]/(cm_rf[1,1]+cm_rf[1,0]),
        2*cm_rf[1,1]/(2*cm_rf[1,1]+cm_rf[0,1]+cm_rf[1,0])
    ]
})

print(comparison)

# 13. Visualization of comparison
plt.figure(figsize=(15, 10))

# Confusion matrices
plt.subplot(2, 2, 1)
sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Isolation Forest Confusion Matrix', fontsize=12, fontweight='bold')

plt.subplot(2, 2, 2)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Random Forest Confusion Matrix', fontsize=12, fontweight='bold')

# Metrics comparison
plt.subplot(2, 2, 3)
metrics = ['False Positives', 'False Negatives', 'F1-Score (Fraud)']
x = np.arange(len(metrics))
width = 0.35

iso_values = [cm_iso[0,1], cm_iso[1,0], 2*cm_iso[1,1]/(2*cm_iso[1,1]+cm_iso[0,1]+cm_iso[1,0])]
rf_values = [cm_rf[0,1], cm_rf[1,0], 2*cm_rf[1,1]/(2*cm_rf[1,1]+cm_rf[0,1]+cm_rf[1,0])]

plt.bar(x - width/2, iso_values, width, label='Isolation Forest', color='blue')
plt.bar(x + width/2, rf_values, width, label='Random Forest', color='green')

plt.xlabel('Metrics')
plt.ylabel('Count / Score')
plt.title('Model Performance Comparison', fontsize=12, fontweight='bold')
plt.xticks(x, metrics)
plt.legend()

# Precision-Recall curves
plt.subplot(2, 2, 4)
# Isolation Forest PR curve
iso_scores = iso_forest.decision_function(X_test_scaled)
precision_iso, recall_iso, _ = precision_recall_curve(y_test, iso_scores)
pr_auc_iso = auc(recall_iso, precision_iso)
plt.plot(recall_iso, precision_iso, label=f'Isolation Forest (AUC = {pr_auc_iso:.3f})')

# Random Forest PR curve
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_proba)
pr_auc_rf = auc(recall_rf, precision_rf)
plt.plot(recall_rf, precision_rf, label=f'Random Forest (AUC = {pr_auc_rf:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves', fontsize=12, fontweight='bold')
plt.legend()

plt.tight_layout()
plt.savefig('./output/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 14. Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15), palette='viridis')
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('./output/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 15. Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Key Findings:")
print(f"1. Isolation Forest False Positives: {cm_iso[0,1]}")
print(f"2. Random Forest False Positives: {cm_rf[0,1]}")
print(f"3. Random Forest reduces false positives by {cm_iso[0,1] - cm_rf[0,1]} ({(cm_iso[0,1] - cm_rf[0,1])/cm_iso[0,1]*100:.1f}%)")
print(f"4. Random Forest maintains good fraud detection (Recall: {cm_rf[1,1]/(cm_rf[1,1]+cm_rf[1,0]):.3f})")
print(f"5. Most important features: {', '.join(feature_importance.head(3)['feature'].tolist())}")

print("\nRecommendation: Use Random Forest for significantly better performance with fewer false positives.")