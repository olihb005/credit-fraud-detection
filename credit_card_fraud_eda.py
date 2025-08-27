# credit_card_fraud_eda.py
# Comprehensive EDA for Credit Card Fraud Detection Dataset

# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Set aesthetic parameters for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 2. Load the dataset
# Ensure the file path is correct. Download from Kaggle and place in your project directory.
df = pd.read_csv('./data/creditcard.csv')

# 3. Initial Data Overview
print("="*50)
print("DATASET SHAPE")
print("="*50)
print(f"Number of transactions: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print("\n")

print("="*50)
print("DATASET INFO")
print("="*50)
df.info()
print("\n")

print("="*50)
print("MISSING VALUES")
print("="*50)
print(df.isnull().sum().max())  # Check if any column has missing values
print("\n")

print("="*50)
print("CLASS DISTRIBUTION (Absolute Numbers)")
print("="*50)
class_dist = df['Class'].value_counts()
print(class_dist)
print("\n")

print("="*50)
print("CLASS DISTRIBUTION (Percentages)")
print("="*50)
class_dist_pct = df['Class'].value_counts(normalize=True) * 100
print(class_dist_pct)
print("\n")

# 4. Detailed Summary Statistics
print("="*50)
print("SUMMARY STATISTICS FOR 'AMOUNT' AND 'TIME'")
print("="*50)
print(df[['Amount', 'Time']].describe())
print("\n")

# 5. Visualizations

# 5.1 Class Distribution Plot
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Class', data=df, palette='viridis')
plt.title('Highly Imbalanced Distribution of Classes', fontsize=16, fontweight='bold')
plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
plt.ylabel('Count')

# Annotate the bars with exact counts and percentages
for p, (count, pct) in zip(ax.patches, zip(class_dist, class_dist_pct)):
    ax.annotate(f'{count}\n({pct:.2f}%)', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                textcoords='offset points')

plt.tight_layout()
plt.savefig('./output/class_distribution.png')  # Save plot to an 'output' folder
plt.show()

# 5.2 Distribution of Transaction Amount by Class
plt.figure(figsize=(12, 6))

# Plot for Legitimate Transactions (Class 0)
plt.subplot(1, 2, 1)
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, kde=True, color='green')
plt.title('Distribution of Transaction Amount\n(Legitimate Transactions)', fontsize=14)
plt.xlabel('Amount ($)')
plt.ylabel('Frequency')
plt.yscale('log')  # Use log scale due to extreme outliers

# Plot for Fraudulent Transactions (Class 1)
plt.subplot(1, 2, 2)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, kde=True, color='red')
plt.title('Distribution of Transaction Amount\n(Fraudulent Transactions)', fontsize=14)
plt.xlabel('Amount ($)')
plt.ylabel('Frequency')
plt.yscale('log')  # Use log scale

plt.tight_layout()
plt.savefig('./output/amount_distribution_by_class.png')
plt.show()

# 5.3 Distribution of Transaction Time by Class
plt.figure(figsize=(12, 6))

# Plot for Legitimate Transactions (Class 0)
plt.subplot(1, 2, 1)
sns.histplot(df[df['Class'] == 0]['Time']/(3600), bins=48, color='green', kde=True) # Convert seconds to hours
plt.title('Transaction Time Distribution\n(Legitimate Transactions)', fontsize=14)
plt.xlabel('Time (Hours since first transaction)')
plt.ylabel('Frequency')

# Plot for Fraudulent Transactions (Class 1)
plt.subplot(1, 2, 2)
sns.histplot(df[df['Class'] == 1]['Time']/(3600), bins=48, color='red', kde=True) # Convert seconds to hours
plt.title('Transaction Time Distribution\n(Fraudulent Transactions)', fontsize=14)
plt.xlabel('Time (Hours since first transaction)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('./output/time_distribution_by_class.png')
plt.show()

# 5.4 Correlation Heatmap (Focus on 'Amount', 'Time', and 'Class')
# Scaling 'Amount' and 'Time' for better correlation analysis
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df_scaled['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Calculate correlations
corr_matrix = df_scaled[['Scaled_Amount', 'Scaled_Time', 'Class']].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create a mask for the upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap: Amount, Time, and Class', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./output/correlation_heatmap.png')
plt.show()

print("="*50)
print("SCRIPT EXECUTION COMPLETE.")
print("Check the './output' folder for generated graphs.")
print("="*50)