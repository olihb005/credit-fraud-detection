"""
Neural Network EDA for Credit Card Fraud Detection
Author: Kaelen (Oliver Hayes-Bradley)
Date: 27/8/25
Objective: Exploratory Data Analysis and Neural Network Training using sklearn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('./data/creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Class distribution
print("\nClass Distribution:")
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"Fraud percentage: {class_counts[1] / len(df) * 100:.4f}%")

# Missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Visualize class distribution
plt.figure(figsize=(8, 6))
plt.bar(['Legitimate (0)', 'Fraudulent (1)'], class_counts.values, color=['green', 'red'])
plt.title('Class Distribution')
plt.ylabel('Count')
plt.savefig('./output/nn_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Amount distribution by class
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df[df['Class']==0]['Amount'], bins=50, color='green', alpha=0.7)
plt.title('Legitimate Transaction Amounts')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.hist(df[df['Class']==1]['Amount'], bins=50, color='red', alpha=0.7)
plt.title('Fraudulent Transaction Amounts')
plt.yscale('log')
plt.tight_layout()
plt.savefig('./output/nn_amount_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Time distribution by class
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df[df['Class']==0]['Time']/3600, bins=48, color='green', alpha=0.7)
plt.title('Legitimate Transaction Times')
plt.xlabel('Hours since first transaction')

plt.subplot(1, 2, 2)
plt.hist(df[df['Class']==1]['Time']/3600, bins=48, color='red', alpha=0.7)
plt.title('Fraudulent Transaction Times')
plt.xlabel('Hours since first transaction')
plt.tight_layout()
plt.savefig('./output/nn_time_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation matrix (only showing a subset for readability)
plt.figure(figsize=(10, 8))
# Select a subset of features for correlation
features_to_correlate = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Time', 'Class']
corr_matrix = df[features_to_correlate].corr()
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(features_to_correlate)), features_to_correlate, rotation=45)
plt.yticks(range(len(features_to_correlate)), features_to_correlate)
plt.title('Correlation Matrix (Selected Features)')
plt.savefig('./output/nn_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Prepare data for modeling
X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")
print(f"Training fraud cases: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
print(f"Test fraud cases: {y_test.sum()} ({y_test.mean()*100:.4f}%)")

# Train Neural Network
print("\n" + "="*60)
print("TRAINING NEURAL NETWORK")
print("="*60)

# Create and train the neural network
nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42,
    verbose=True
)

nn_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = nn_model.predict(X_test_scaled)
y_proba = nn_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Calculate metrics
from sklearn.metrics import f1_score, precision_score, recall_score
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\nMetrics:")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Legitimate', 'Fraudulent'])
plt.yticks(tick_marks, ['Legitimate', 'Fraudulent'])

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('./output/nn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('./output/nn_precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Learning curve
plt.figure(figsize=(10, 6))
plt.plot(nn_model.loss_curve_, label='Training Loss')
plt.plot(nn_model.validation_scores_, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.savefig('./output/nn_learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the model
joblib.dump(nn_model, './models/neural_network_eda.pkl')
joblib.dump(scaler, './models/nn_scaler.pkl')

print("\n" + "="*60)
print("NEURAL NETWORK EDA COMPLETE")
print("="*60)
print("Model saved as './models/neural_network_eda.pkl'")
print("Scaler saved as './models/nn_scaler.pkl'")
print(f"Final F1 Score: {f1:.4f}")