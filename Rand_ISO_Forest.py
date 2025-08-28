# =============================================================================
# Credit Card Fraud Detection - Final Model Implementation
# Author: OHB
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
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create output directory
import os
if not os.path.exists('./output'):
    os.makedirs('./output')
if not os.path.exists('./models'):
    os.makedirs('./models')

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class FraudDetectionModel:
    def __init__(self):
        self.iso_forest = None
        self.random_forest = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load and prepare the dataset."""
        self.df = pd.read_csv(filepath)
        self.X = self.df.drop('Class', axis=1)
        self.y = self.df['Class']
        self.feature_names = self.X.columns.tolist()
        
        print(f"Data loaded: {self.X.shape}")
        print(f"Fraud cases: {self.y.sum()} ({self.y.mean()*100:.4f}%)")
        
    def split_and_scale(self):
        """Split data and scale features."""
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nData split:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Training fraud: {self.y_train.sum()}")
        print(f"Test fraud: {self.y_test.sum()}")
        
    def train_isolation_forest(self):
        """Train Isolation Forest model."""
        print("\nTraining Isolation Forest...")
        
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.0017,  # Match actual fraud rate
            random_state=42,
            n_jobs=-1
        )
        
        self.iso_forest.fit(self.X_train_scaled)
        print("Isolation Forest trained successfully!")
        
    def train_random_forest(self):
        """Train Random Forest model."""
        print("\nTraining Random Forest...")
        
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        self.random_forest.fit(self.X_train_scaled, self.y_train)
        print("Random Forest trained successfully!")
        
    def evaluate_models(self):
        """Evaluate both models and compare performance."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Isolation Forest predictions
        y_pred_iso = self.iso_forest.predict(self.X_test_scaled)
        y_pred_iso = np.where(y_pred_iso == -1, 1, 0)
        
        # Random Forest predictions
        y_pred_rf = self.random_forest.predict(self.X_test_scaled)
        y_proba_rf = self.random_forest.predict_proba(self.X_test_scaled)[:, 1]
        
        # Confusion matrices
        cm_iso = confusion_matrix(self.y_test, y_pred_iso)
        cm_rf = confusion_matrix(self.y_test, y_pred_rf)
        
        # Results summary
        results = {
            'Isolation Forest': {
                'False Positives': cm_iso[0, 1],
                'False Negatives': cm_iso[1, 0],
                'Precision': cm_iso[1, 1] / (cm_iso[1, 1] + cm_iso[0, 1]),
                'Recall': cm_iso[1, 1] / (cm_iso[1, 1] + cm_iso[1, 0]),
                'F1-Score': 2 * cm_iso[1, 1] / (2 * cm_iso[1, 1] + cm_iso[0, 1] + cm_iso[1, 0])
            },
            'Random Forest': {
                'False Positives': cm_rf[0, 1],
                'False Negatives': cm_rf[1, 0],
                'Precision': cm_rf[1, 1] / (cm_rf[1, 1] + cm_rf[0, 1]),
                'Recall': cm_rf[1, 1] / (cm_rf[1, 1] + cm_rf[1, 0]),
                'F1-Score': 2 * cm_rf[1, 1] / (2 * cm_rf[1, 1] + cm_rf[0, 1] + cm_rf[1, 0])
            }
        }
        
        # Print results
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  False Positives: {metrics['False Positives']}")
            print(f"  False Negatives: {metrics['False Negatives']}")
            print(f"  Precision (Fraud): {metrics['Precision']:.4f}")
            print(f"  Recall (Fraud): {metrics['Recall']:.4f}")
            print(f"  F1-Score (Fraud): {metrics['F1-Score']:.4f}")
        
        # Calculate improvement
        fp_improvement = results['Isolation Forest']['False Positives'] - results['Random Forest']['False Positives']
        fp_improvement_pct = (fp_improvement / results['Isolation Forest']['False Positives']) * 100
        
        print(f"\nRandom Forest Improvement:")
        print(f"  False Positives reduced by: {fp_improvement} ({fp_improvement_pct:.1f}%)")
        print(f"  Maintains fraud detection: {results['Random Forest']['Recall']:.3f} recall")
        
        return results, cm_iso, cm_rf, y_proba_rf
        
    def visualize_results(self, cm_iso, cm_rf, y_proba_rf):
        """Create visualizations for model comparison."""
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
        
        # Precision-Recall curves
        plt.subplot(2, 2, 3)
        # Isolation Forest
        iso_scores = self.iso_forest.decision_function(self.X_test_scaled)
        precision_iso, recall_iso, _ = precision_recall_curve(self.y_test, iso_scores)
        pr_auc_iso = auc(recall_iso, precision_iso)
        plt.plot(recall_iso, precision_iso, label=f'Isolation Forest (AUC = {pr_auc_iso:.3f})')
        
        # Random Forest
        precision_rf, recall_rf, _ = precision_recall_curve(self.y_test, y_proba_rf)
        pr_auc_rf = auc(recall_rf, precision_rf)
        plt.plot(recall_rf, precision_rf, label=f'Random Forest (AUC = {pr_auc_rf:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves', fontsize=12, fontweight='bold')
        plt.legend()
        
        # Feature importance
        plt.subplot(2, 2, 4)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.random_forest.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10), palette='viridis')
        plt.title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
        plt.xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('./output/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_models(self):
        """Save trained models for future use."""
        print("\nSaving models...")
        
        # Save Isolation Forest
        joblib.dump({
            'model': self.iso_forest,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, './models/isolation_forest.pkl')
        
        # Save Random Forest
        joblib.dump({
            'model': self.random_forest,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, './models/random_forest.pkl')
        
        print("Models saved successfully!")
        
    def load_models(self, model_type='random_forest'):
        """Load a saved model."""
        print(f"\nLoading {model_type} model...")
        
        if model_type == 'isolation_forest':
            data = joblib.load('./models/isolation_forest.pkl')
            self.iso_forest = data['model']
        elif model_type == 'random_forest':
            data = joblib.load('./models/random_forest.pkl')
            self.random_forest = data['model']
        
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        print(f"{model_type} model loaded successfully!")
        
    def predict(self, X, model_type='random_forest'):
        """Make predictions using the specified model."""
        X_scaled = self.scaler.transform(X)
        
        if model_type == 'isolation_forest':
            predictions = self.iso_forest.predict(X_scaled)
            return np.where(predictions == -1, 1, 0)
        elif model_type == 'random_forest':
            return self.random_forest.predict(X_scaled)

def main():
    """Main execution function."""
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - FINAL MODEL IMPLEMENTATION")
    print("="*60)
    
    # Initialize model
    detector = FraudDetectionModel()
    
    # Load and prepare data
    detector.load_data('./data/creditcard.csv')
    detector.split_and_scale()
    
    # Train models
    detector.train_isolation_forest()
    detector.train_random_forest()
    
    # Evaluate models
    results, cm_iso, cm_rf, y_proba_rf = detector.evaluate_models()
    
    # Visualize results
    detector.visualize_results(cm_iso, cm_rf, y_proba_rf)
    
    # Save models
    detector.save_models()
    
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)
    print("Random Forest is the superior model with:")
    print(f"- {results['Random Forest']['False Positives']} false positives (vs {results['Isolation Forest']['False Positives']})")
    print(f"- {results['Random Forest']['Recall']:.3f} recall for fraud detection")
    print(f"- {results['Random Forest']['F1-Score']:.3f} F1-score")
    print("\nUse Random Forest for production deployment.")

if __name__ == "__main__":
    main()