import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

def perform_feature_engineering(filepath, data_dir, model_dir):
    print(f"Reading cleaned data from {filepath}")
    df = pd.read_csv(filepath)
    
    # 1. Target separation and encoding
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")
        
    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 2. Train/Test Split (stratified)
    # Perform this BEFORE encoding/scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train/Test split complete.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Identify numerical and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # 3. Create Preprocessing Pipeline (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # 4. Fit and Transform
    print("Fitting preprocessor and transforming data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding for later use (like Feature Importance)
    num_names = numeric_features
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_names, cat_names])
    print(f"Total features after encoding: {len(feature_names)}")
    
    # Convert arrays back to DataFrames for easier handling in modeling and SHAP
    X_train_processed = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # 5. Save artifacts
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save transformed data
    X_train_processed.to_csv(os.path.join(data_dir, 'X_train.csv'), index=False)
    X_test_processed.to_csv(os.path.join(data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)
    
    # Save preprocessor and feature names
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.joblib'))
    
    print("Feature Engineering completed successfully.")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'cleaned_churn_data.csv')
    data_dir = os.path.join(base_path, 'data')
    model_dir = os.path.join(base_path, 'models')
    
    perform_feature_engineering(data_path, data_dir, model_dir)
