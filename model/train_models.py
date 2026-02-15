import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the Bank Marketing dataset
    """
    # Load dataset - try comma separator first, then semicolon
    try:
        df = pd.read_csv(filepath, sep=',')
    except:
        df = pd.read_csv(filepath, sep=';')
    
    # Print columns to debug
    print(f"Columns in dataset: {df.columns.tolist()}")
    
    # Handle missing values if any
    df = df.dropna()
    
    # Check if target column exists (could be 'deposit' or 'y')
    if 'y' in df.columns:
        target_col = 'y'
    elif 'deposit' in df.columns:
        target_col = 'deposit'
    else:
        raise ValueError("Target column not found. Expected 'deposit' or 'y'")
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 
                          'loan', 'contact', 'month', 'poutcome']
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    label_encoders['target'] = le_target
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y, label_encoders


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Calculate all 6 evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # AUC requires probability scores
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train all 6 models and evaluate them
    """
    results = {}
    models_dict = {}
    
    # Scale features for models that benefit from it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    results['Logistic Regression'] = evaluate_model(y_test, y_pred, y_pred_proba)
    models_dict['Logistic Regression'] = lr_model
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    
    # 2. Decision Tree
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    y_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    results['Decision Tree'] = evaluate_model(y_test, y_pred, y_pred_proba)
    models_dict['Decision Tree'] = dt_model
    joblib.dump(dt_model, 'decision_tree_model.pkl')
    
    # 3. K-Nearest Neighbors
    print("Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    y_pred = knn_model.predict(X_test_scaled)
    y_pred_proba = knn_model.predict_proba(X_test_scaled)[:, 1]
    results['kNN'] = evaluate_model(y_test, y_pred, y_pred_proba)
    models_dict['kNN'] = knn_model
    joblib.dump(knn_model, 'knn_model.pkl')
    
    # 4. Naive Bayes
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    y_pred = nb_model.predict(X_test_scaled)
    y_pred_proba = nb_model.predict_proba(X_test_scaled)[:, 1]
    results['Naive Bayes'] = evaluate_model(y_test, y_pred, y_pred_proba)
    models_dict['Naive Bayes'] = nb_model
    joblib.dump(nb_model, 'naive_bayes_model.pkl')
    
    # 5. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    results['Random Forest (Ensemble)'] = evaluate_model(y_test, y_pred, y_pred_proba)
    models_dict['Random Forest (Ensemble)'] = rf_model
    joblib.dump(rf_model, 'random_forest_model.pkl')
    
    # 6. XGBoost
    print("Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, max_depth=6, 
                             use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    results['XGBoost (Ensemble)'] = evaluate_model(y_test, y_pred, y_pred_proba)
    models_dict['XGBoost (Ensemble)'] = xgb_model
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    return results, models_dict


def print_results(results):
    """
    Print results in a formatted table
    """
    print("\n" + "="*100)
    print("MODEL EVALUATION RESULTS")
    print("="*100)
    print(f"{'Model':<30} {'Accuracy':<12} {'AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MCC':<12}")
    print("-"*100)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['Accuracy']:<12.4f} {metrics['AUC']:<12.4f} "
              f"{metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f} "
              f"{metrics['F1']:<12.4f} {metrics['MCC']:<12.4f}")
    
    print("="*100)


if __name__ == "__main__":
    # Note: Download the dataset from Kaggle first
    # Dataset: Bank Marketing Data Set
    # URL: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data
    
    print("Loading and preprocessing data...")
    try:
        # Try to load the dataset (looking in parent directory)
        X, y, label_encoders = load_and_preprocess_data('C:\\Users\\prakh\\Downloads\\ML_Test\\bank-full.csv')
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Target distribution:\n{pd.Series(y).value_counts()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train and evaluate all models
        print("\nTraining models...")
        results, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Print results
        print_results(results)
        
        # Save results to CSV for easy reference
        results_df = pd.DataFrame(results).T
        results_df.to_csv('model_results.csv')
        print("\nResults saved to 'model_results.csv'")
        
        # Save label encoders
        joblib.dump(label_encoders, 'label_encoders.pkl')
        print("Label encoders saved!")
        
        print("\n✓ All models trained and saved successfully!")
        
    except FileNotFoundError:
        print("\nError: Dataset file 'bank-full.csv' not found!")
        print("Please download the Bank Marketing dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data")
        print("Place 'bank-full.csv' in the project root directory.")
