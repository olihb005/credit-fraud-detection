# Credit Card Fraud Detection

## Overview
This project implements and compares two machine learning approaches for credit card fraud detection: Isolation Forest (unsupervised anomaly detection) and Random Forest (supervised classification). The goal is to identify fraudulent transactions while minimizing false positives.

## Results
### Isolation Forest Model
- **Recall**: 0.92
- **Precision**: 0.15
- **F1-Score**: 0.26
- **False Positives**: ~5,000-10,000

*This high recall, low precision profile is excellent for a first-pass anomaly detector, correctly identifying 92% of fraudulent transactions.*

### Random Forest Model
- **Recall**: 0.75
- **Precision**: 0.85
- **F1-Score**: 0.80 (with threshold optimization)
- **False Positives**: ~200-500

*The Random Forest model achieved a 97% reduction in false positives compared to Isolation Forest while maintaining good fraud detection capability.*

![Model Comparison](./output/model_comparison.png)

## Technical Specifications
### Libraries Used
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `imbalanced-learn` - Handling imbalanced datasets
- `jupyter` - Interactive computing environment
- `graphviz` - Graph visualization software

### Dataset
- **Source**: Kaggle Credit Card Fraud Dataset
- **Size**: 284,807 transactions
- **Features**: 30 (V1-V28 PCA components, Time, Amount)
- **Class Imbalance**: 0.17% fraudulent transactions

## Project Structure
