# =============================================================================
# Credit Card Fraud Detection - Tree Visualization
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import plot_tree, export_graphviz
from sklearn.preprocessing import StandardScaler
import graphviz
import warnings
warnings.filterwarnings('ignore')

# Create output directory
import os
if not os.path.exists('./output'):
    os.makedirs('./output')
if not os.path.exists('./output/trees'):
    os.makedirs('./output/trees')

# Set style
plt.rcParams['figure.figsize'] = (20, 10)  # EDIT LINE 25: Change figure size
plt.rcParams['font.size'] = 10  # EDIT LINE 26: Change font size

def load_and_prepare_data():
    print("Loading and preparing data...")
    
    df = pd.read_csv('./data/creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data loaded: {X_train.shape}")
    return X_train_scaled, y_train, X_test_scaled, y_test, X.columns

def train_models(X_train_scaled, y_train):
    print("\nTraining models...")
    
    rf = RandomForestClassifier(
        n_estimators=100,  # EDIT LINE 50: Change number of trees
        class_weight='balanced',
        max_depth=8,  # EDIT LINE 52: Change max depth
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    iso_forest = IsolationForest(
        n_estimators=100,  # EDIT LINE 60: Change number of trees
        contamination=0.0017,  # EDIT LINE 61: Change contamination rate
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)
    
    print("Models trained successfully!")
    return rf, iso_forest

def visualize_random_forest_tree(rf, feature_names):
    print("\nVisualizing Random Forest tree...")
    
    tree = rf.estimators_[0]  # EDIT LINE 77: Change tree index (0 to 99)
    
    fig, ax = plt.subplots(figsize=(25, 15))  # EDIT LINE 79: Change figure size
    
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=['Legitimate', 'Fraudulent'],  # EDIT LINE 83: Change class names
        filled=True,
        rounded=True,
        fontsize=8,  # EDIT LINE 86: Change font size
        max_depth=3,  # EDIT LINE 87: CHANGE TREE DEPTH (KEY EDIT)
        ax=ax
    )
    
    plt.title("Random Forest - Sample Decision Tree", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./output/trees/random_forest_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed version
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=['Legitimate', 'Fraudulent'],  # EDIT LINE 100: Change class names
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3  # EDIT LINE 104: CHANGE TREE DEPTH (KEY EDIT)
    )
    
    graph = graphviz.Source(dot_data)
    graph.render("./output/trees/random_forest_tree_detailed", format="png", cleanup=True)
    
    print("Random Forest tree visualizations saved!")

def visualize_isolation_forest_tree(iso_forest, feature_names):
    print("\nVisualizing Isolation Forest tree...")
    
    tree = iso_forest.estimators_[0]  # EDIT LINE 117: Change tree index (0 to 99)
    
    fig, ax = plt.subplots(figsize=(25, 15))  # EDIT LINE 119: Change figure size
    
    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=8,  # EDIT LINE 125: Change font size
        max_depth=3,  # EDIT LINE 126: CHANGE TREE DEPTH (KEY EDIT)
        ax=ax
    )
    
    plt.title("Isolation Forest - Sample Isolation Tree", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./output/trees/isolation_forest_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed version
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3  # EDIT LINE 143: CHANGE TREE DEPTH (KEY EDIT)
    )
    
    graph = graphviz.Source(dot_data)
    graph.render("./output/trees/isolation_forest_tree_detailed", format="png", cleanup=True)
    
    print("Isolation Forest tree visualizations saved!")

def create_comparison_visualization(rf, iso_forest, feature_names):
    print("\nCreating comparison visualization...")
    
    rf_tree = rf.estimators_[0]  # EDIT LINE 156: Change tree index
    iso_tree = iso_forest.estimators_[0]  # EDIT LINE 157: Change tree index
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))  # EDIT LINE 159: Change figure size
    
    # Random Forest tree
    plot_tree(
        rf_tree,
        feature_names=feature_names,
        class_names=['Legitimate', 'Fraudulent'],  # EDIT LINE 164: Change class names
        filled=True,
        rounded=True,
        fontsize=6,  # EDIT LINE 167: Change font size
        max_depth=2,  # EDIT LINE 168: CHANGE TREE DEPTH (KEY EDIT)
        ax=ax1
    )
    ax1.set_title("Random Forest Decision Tree", fontsize=14, fontweight='bold')
    
    # Isolation Forest tree
    plot_tree(
        iso_tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=6,  # EDIT LINE 178: Change font size
        max_depth=2,  # EDIT LINE 179: CHANGE TREE DEPTH (KEY EDIT)
        ax=ax2
    )
    ax2.set_title("Isolation Forest Tree", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/trees/tree_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison visualization saved!")

def main():
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - TREE VISUALIZATION")
    print("="*60)
    
    X_train_scaled, y_train, X_test_scaled, y_test, feature_names = load_and_prepare_data()
    rf, iso_forest = train_models(X_train_scaled, y_train)
    
    visualize_random_forest_tree(rf, feature_names)
    visualize_isolation_forest_tree(iso_forest, feature_names)
    create_comparison_visualization(rf, iso_forest, feature_names)
    
    print("\n" + "="*60)
    print("TREE VISUALIZATION COMPLETE")
    print("="*60)
    print("Check './output/trees/' directory for generated images")

if __name__ == "__main__":
    main()