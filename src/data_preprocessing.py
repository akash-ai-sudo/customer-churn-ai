import pandas as pd
import numpy as np
import os

def load_and_clean_data(filepath):
    """
    Loads dataset, inspects it, cleans the data, and handles missing values.
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    print("\n--- 1. Data Inspection ---")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    print("\n--- 2. Data Cleaning & Missing Value Handling ---")
    
    # Drop customerID (not useful for prediction)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        print("-> Dropped 'customerID' column.")
    
    # TotalCharges contains " " for new customers, which causes it to be parsed as object.
    # Convert to numeric, turning " " into NaN.
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        print("-> Converted 'TotalCharges' to numeric. Coerced empty strings to NaN.")
    
    # Handle Missing values
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    
    if total_missing > 0:
        print(f"-> Found {total_missing} missing values:")
        print(missing_summary[missing_summary > 0])
        # We generally have merely 11 missing values in TotalCharges, so we drop them.
        df = df.dropna()
        print(f"-> Missing values dropped. Current total missing: {df.isnull().sum().sum()}")
    else:
        print("-> No missing values found.")
        
    print("\n--- Cleaned Data Summary ---")
    print(f"Cleaned dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Define absolute paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'Telco-Customer-Churn.csv')
    
    # Run pipeline
    cleaned_df = load_and_clean_data(data_path)
    
    # Save the cleaned dataset
    out_path = os.path.join(base_path, 'data', 'cleaned_churn_data.csv')
    cleaned_df.to_csv(out_path, index=False)
    print(f"\nCleaned data successfully saved to: {out_path}")
