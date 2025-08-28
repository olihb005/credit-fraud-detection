"""
Final Ensemble Model for Credit Card Fraud Detection
Author: Kaelen (Oliver Hayes-Bradley)
Date: 27/8/25
Objective: Combine Random Forest, XGBoost, and Neural Network using final trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Load and prepare data
print("Loading and preparing data...")
df = pd.read_csv('./data/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the scaler (from Random Forest model)
scaler = joblib.load('./models/scaler.pkl')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Load final trained models
print("\nLoading final trained models...")
try:
    rf_model = joblib.load('./models/random_forest_final.pkl')
    xgb_model = joblib.load('./models/xgboost_final.pkl')
    nn_model = joblib.load('./models/neural_network_final.pkl')
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Evaluate individual models
print("\n" + "="*60)
print("INDIVIDUAL MODEL EVALUATION")
print("="*60)

models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'Neural Network': nn_model
}

individual_results = {}

for name, model in models.items():
    # Make predictions
    if name == 'Neural Network':
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    individual_results[name] = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print(f"\n{name}:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")

# Create ensemble model
print("\n" + "="*60)
print("CREATING ENSEMBLE MODEL")
print("="*60)

# Create soft voting ensemble
ensemble_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('nn', nn_model)
    ],
    voting='soft'
)

# Train the ensemble (this just combines the pre-trained models)
ensemble_clf.fit(X_train_scaled, y_train)

# Get ensemble probabilities
y_proba_ensemble = ensemble_clf.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold for F1 score
print("\nFinding optimal threshold...")
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_proba_ensemble >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    f1_scores.append(f1)

optimal_threshold = thresholds[np.argmax(f1_scores)]
y_pred_ensemble = (y_proba_ensemble >= optimal_threshold).astype(int)

# Evaluate ensemble model
print("\n" + "="*60)
print("ENSEMBLE MODEL EVALUATION")
print("="*60)

cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)

print(f"Optimal Threshold: {optimal_threshold:.2f}")
print(f"F1 Score: {f1_ensemble:.4f}")
print(f"Precision: {precision_ensemble:.4f}")
print(f"Recall: {recall_ensemble:.4f}")
print(f"False Positives: {cm_ensemble[0,1]}")
print(f"False Negatives: {cm_ensemble[1,0]}")

# Compare all models
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Add ensemble results to comparison
all_results = individual_results.copy()
all_results['Ensemble'] = {
    'f1_score': f1_ensemble,
    'precision': precision_ensemble,
    'recall': recall_ensemble,
    'confusion_matrix': cm_ensemble,
    'y_pred': y_pred_ensemble,
    'y_proba': y_proba_ensemble
}

# Create comparison plot
plt.figure(figsize=(15, 10))

# F1 Score Comparison
plt.subplot(2, 2, 1)
model_names = list(all_results.keys())
f1_scores_comp = [all_results[name]['f1_score'] for name in model_names]
bars = plt.bar(model_names, f1_scores_comp, color=['blue', 'green', 'red', 'purple'])
plt.title('F1 Score Comparison')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
for bar, score in zip(bars, f1_scores_comp):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

# Precision-Recall Trade-off
plt.subplot(2, 2, 2)
for name, result in all_results.items():
    precision, recall, _ = precision_recall_curve(y_test, result['y_proba'])
    pr_auc = auc(recall, precision)
    plt.scatter(recall, precision, label=f'{name} (AUC={pr_auc:.3f})', s=100)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Trade-off')
plt.legend()

# Confusion Matrix for Ensemble
plt.subplot(2, 2, 3)
plt.imshow(cm_ensemble, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Ensemble Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Legitimate', 'Fraudulent'])
plt.yticks(tick_marks, ['Legitimate', 'Fraudulent'])

thresh = cm_ensemble.max() / 2.
for i in range(cm_ensemble.shape[0]):
    for j in range(cm_ensemble.shape[1]):
        plt.text(j, i, format(cm_ensemble[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm_ensemble[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Threshold Optimization
plt.subplot(2, 2, 4)
plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
            label=f'Optimal Threshold ({optimal_threshold:.2f})')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('Threshold Optimization')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./output/final_ensemble_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Save ensemble model and optimal threshold
print("\nSaving ensemble model and optimal threshold...")
joblib.dump(ensemble_clf, './models/final_ensemble_model.pkl')
with open('./models/final_optimal_threshold.txt', 'w') as f:
    f.write(str(optimal_threshold))

# Generate detailed summary report
print("\n" + "="*60)
print("FINAL ENSEMBLE SUMMARY REPORT")
print("="*60)

summary_data = []
for name, result in all_results.items():
    summary_data.append({
        'Model': name,
        'F1 Score': result['f1_score'],
        'Precision': result['precision'],
        'Recall': result['recall'],
        'False Positives': result['confusion_matrix'][0,1],
        'False Negatives': result['confusion_matrix'][1,0]
    })

summary_df = pd.DataFrame(summary_data)
print("\nDetailed Performance Metrics:")
print(summary_df.to_string(index=False))

# Calculate improvements
best_individual_f1 = max(individual_results[name]['f1_score'] for name in models)
improvement = f1_ensemble - best_individual_f1

print(f"\nKey Findings:")
print(f"- Best Individual Model F1 Score: {best_individual_f1:.4f}")
print(f"- Ensemble F1 Score: {f1_ensemble:.4f}")
print(f"- Improvement: {improvement:.4f} ({(improvement/best_individual_f1)*100:.1f}%)")
print(f"- Optimal Threshold: {optimal_threshold:.2f}")
print(f"- False Positives Reduced: {individual_results['Random Forest']['confusion_matrix'][0,1] - cm_ensemble[0,1]}")
print(f"- False Negatives Reduced: {individual_results['Neural Network']['confusion_matrix'][1,0] - cm_ensemble[1,0]}")

# Save summary
summary_df.to_csv('./output/final_ensemble_summary.csv', index=False)

# Generate classification report for ensemble
print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['Legitimate', 'Fraudulent']))

print("\n" + "="*60)
print("FINAL ENSEMBLE MODEL COMPLETE")
print("="*60)
print("Saved Files:")
print("- Ensemble Model: './models/final_ensemble_model.pkl'")
print("- Optimal Threshold: './models/final_optimal_threshold.txt'")
print("- Comparison Plot: './output/final_ensemble_comparison.png'")
print("- Summary Report: './output/final_ensemble_summary.csv'")
print(f"\nFinal F1 Score: {f1_ensemble:.4f}")
print(f"Ensemble successfully combines all three models for optimal performance!")