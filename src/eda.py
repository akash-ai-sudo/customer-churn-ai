import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(filepath, report_dir):
    """
    Reads the cleaned data, produces summary statistics, 
    and saves EDA and correlation visualizations to the reports directory.
    """
    print(f"Reading cleaned dataset from {filepath}")
    df = pd.read_csv(filepath)
    
    os.makedirs(report_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\n--- Summary Statistics ---")
    print(df.describe(include='all'))
    
    print("\n--- Generating Visualizations ---")
    # 1. Target Variable Distribution (Churn rate)
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Churn', palette='viridis')
    plt.title('Customer Churn Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'churn_distribution.png'))
    plt.close()
    
    # 2. Churn by Contract Type
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='viridis')
    plt.title('Churn by Contract Type')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'churn_by_contract.png'))
    plt.close()
    
    # 3. Monthly Charges Distribution by Churn
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple="stack", palette='viridis', kde=True)
    plt.title('Monthly Charges Distribution by Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'monthly_charges_churn.png'))
    plt.close()
    
    # 4. Correlation Heatmap (numeric features + mapped target)
    numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
    if 'Churn' in df.columns:
        numeric_df['Churn_Num'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'correlation_matrix.png'))
    plt.close()
    
    print(f"EDA completed. 4 visualization images saved to {report_dir}\\")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'cleaned_churn_data.csv')
    report_path = os.path.join(base_path, 'reports')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Run data_preprocessing.py first.")
    else:
        perform_eda(data_path, report_path)
