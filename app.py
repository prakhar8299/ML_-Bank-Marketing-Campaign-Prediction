"""
Bank Marketing Classification - Streamlit Web Application
Interactive ML model deployment for prediction and evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 Bank Marketing Campaign Prediction")
st.markdown("""
This application predicts whether a client will subscribe to a term deposit based on bank marketing campaign data.
Upload your test dataset and select a model to see predictions and evaluation metrics.
""")

# Model information
MODEL_INFO = {
    'Logistic Regression': 'model/logistic_regression_model.pkl',
    'Decision Tree': 'model/decision_tree_model.pkl',
    'kNN': 'model/knn_model.pkl',
    'Naive Bayes': 'model/naive_bayes_model.pkl',
    'Random Forest (Ensemble)': 'model/random_forest_model.pkl',
    'XGBoost (Ensemble)': 'model/xgboost_model.pkl'
}

SCALED_MODELS = ['Logistic Regression', 'kNN', 'Naive Bayes']


@st.cache_resource
def load_model(model_name):
    """Load the selected model"""
    model_path = MODEL_INFO[model_name]
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_scaler():
    """Load the scaler"""
    scaler_path = 'model/scaler.pkl'
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


def preprocess_data(df, has_target=True):
    """
    Preprocess the uploaded dataset
    """
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 
                          'loan', 'contact', 'month', 'poutcome']
    
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Handle target variable if present
    if has_target and 'deposit' in df_processed.columns:
        le_target = LabelEncoder()
        y = le_target.fit_transform(df_processed['deposit'])
        X = df_processed.drop('deposit', axis=1)
        return X, y
    else:
        return df_processed, None


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate all evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred):
    """
    Create confusion matrix visualization
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], 
                yticklabels=['No', 'Yes'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    return fig


# Sidebar for model selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Classification Model:",
    list(MODEL_INFO.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About the Models
- **Logistic Regression**: Linear model for binary classification
- **Decision Tree**: Tree-based model with interpretable rules
- **kNN**: Instance-based learning algorithm
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting ensemble method
""")

# Main content area
st.header("📤 Upload Test Dataset")
st.markdown("Upload a CSV file with bank marketing data. The file should contain the same features as the training data.")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            if len(df.columns) == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',')
        except:
            uploaded_file.seek(0)  
            df = pd.read_csv(uploaded_file, sep=',')
        df.columns = df.columns.str.strip()
        st.success(f"✓ File uploaded successfully! Shape: {df.shape}")
        
        # Show data preview
        with st.expander("📊 View Data Preview"):
            st.dataframe(df.head(10))
            st.write(f"**Columns:** {', '.join(df.columns)}")
        
        # Check if target column exists
        has_target = 'deposit' in df.columns
        
        if has_target:
            st.info("Target column 'deposit' found. Evaluation metrics will be calculated.")
        else:
            st.warning("Target column 'deposit' not found. Only predictions will be shown.")
        
        # Preprocess data
        X, y_true = preprocess_data(df, has_target)
        
        # Load model and scaler
        model = load_model(selected_model)
        scaler = load_scaler()
        
        if model is None:
            st.error(f"Model '{selected_model}' not found. Please train the models first.")
        else:
            st.success(f"✓ Model '{selected_model}' loaded successfully!")
            
            # Apply scaling if needed
            if selected_model in SCALED_MODELS and scaler is not None:
                X_processed = scaler.transform(X)
            else:
                X_processed = X
            
            # Make predictions
            y_pred = model.predict(X_processed)
            
            try:
                y_pred_proba = model.predict_proba(X_processed)[:, 1]
            except:
                y_pred_proba = None
            
            # Display results
            st.header("📈 Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Predictions Summary")
                pred_df = pd.DataFrame({
                    'Prediction': ['No', 'Yes'],
                    'Count': [np.sum(y_pred == 0), np.sum(y_pred == 1)],
                    'Percentage': [
                        f"{100*np.sum(y_pred == 0)/len(y_pred):.2f}%",
                        f"{100*np.sum(y_pred == 1)/len(y_pred):.2f}%"
                    ]
                })
                st.dataframe(pred_df, hide_index=True)
            
            with col2:
                if has_target:
                    st.subheader("Actual Distribution")
                    actual_df = pd.DataFrame({
                        'Actual': ['No', 'Yes'],
                        'Count': [np.sum(y_true == 0), np.sum(y_true == 1)],
                        'Percentage': [
                            f"{100*np.sum(y_true == 0)/len(y_true):.2f}%",
                            f"{100*np.sum(y_true == 1)/len(y_true):.2f}%"
                        ]
                    })
                    st.dataframe(actual_df, hide_index=True)
            
            # Evaluation metrics
            if has_target:
                st.subheader("🎯 Evaluation Metrics")
                
                metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)
                
                col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
                col3.metric("Precision", f"{metrics['Precision']:.4f}")
                col4.metric("Recall", f"{metrics['Recall']:.4f}")
                col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                col6.metric("MCC Score", f"{metrics['MCC']:.4f}")
                
                # Confusion Matrix
                st.subheader("📊 Confusion Matrix")
                fig = plot_confusion_matrix(y_true, y_pred)
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("📋 Classification Report")
                report = classification_report(y_true, y_pred, 
                                              target_names=['No', 'Yes'],
                                              output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"))
            
            # Download predictions
            st.subheader("💾 Download Predictions")
            result_df = df.copy()
            result_df['Predicted_Deposit'] = ['yes' if p == 1 else 'no' for p in y_pred]
            
            if y_pred_proba is not None:
                result_df['Prediction_Probability'] = y_pred_proba
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{selected_model.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format with semicolon (;) separator.")

else:
    st.info("👆 Please upload a CSV file to get started.")
    
    # Show expected format
    st.markdown("### Expected CSV Format")
    st.markdown("""
    The CSV file should have the following columns (semicolon-separated):
    - age, job, marital, education, default, balance, housing, loan, contact, day, month, 
      duration, campaign, pdays, previous, poutcome, deposit (optional for evaluation)
    """)
    
    # Sample data
    sample_data = {
        'age': [30, 35, 45],
        'job': ['admin.', 'technician', 'services'],
        'marital': ['married', 'single', 'married'],
        'education': ['secondary', 'secondary', 'secondary'],
        'default': ['no', 'no', 'no'],
        'balance': [1000, 2000, 1500],
        'housing': ['yes', 'no', 'yes'],
        'loan': ['no', 'no', 'yes'],
        'contact': ['cellular', 'cellular', 'cellular'],
        'day': [15, 20, 10],
        'month': ['may', 'jun', 'jul'],
        'duration': [200, 300, 250],
        'campaign': [1, 2, 1],
        'pdays': [-1, -1, 90],
        'previous': [0, 0, 1],
        'poutcome': ['unknown', 'unknown', 'success'],
        'deposit': ['no', 'yes', 'no']
    }
    
    st.markdown("### Sample Data")
    st.dataframe(pd.DataFrame(sample_data))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Bank Marketing Campaign Prediction System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
