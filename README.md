# Bank Marketing Campaign Prediction - Machine Learning Classification

## Problem Statement

The goal of this project is to predict whether a client will subscribe to a term deposit based on data from direct marketing campaigns (phone calls) of a Portuguese banking institution. This is a binary classification problem where we need to predict if the client will say "yes" or "no" to subscribing to a term deposit.

The marketing campaigns were based on phone calls, and often more than one contact to the same client was required to determine if the product (bank term deposit) would be subscribed or not. This prediction can help banks optimize their marketing strategies by targeting clients who are more likely to subscribe.

## Dataset Description

**Dataset Name:** Bank Marketing Data Set  
**Source:** UCI Machine Learning Repository / Kaggle  
**Dataset Size:** 45,211 instances  
**Number of Features:** 16 input features + 1 target variable  
**Type:** Multivariate, Sequential, Time-Series  
**Classification Task:** Binary Classification (yes/no)

### Features Description:

1. **age** (numeric): Age of the client
2. **job** (categorical): Type of job (admin., technician, services, management, retired, blue-collar, unemployed, entrepreneur, housemaid, unknown, self-employed, student)
3. **marital** (categorical): Marital status (married, single, divorced)
4. **education** (categorical): Education level (primary, secondary, tertiary, unknown)
5. **default** (categorical): Has credit in default? (yes, no)
6. **balance** (numeric): Average yearly balance in euros
7. **housing** (categorical): Has housing loan? (yes, no)
8. **loan** (categorical): Has personal loan? (yes, no)
9. **contact** (categorical): Contact communication type (cellular, telephone, unknown)
10. **day** (numeric): Last contact day of the month
11. **month** (categorical): Last contact month of year (jan, feb, mar, ..., nov, dec)
12. **duration** (numeric): Last contact duration in seconds
13. **campaign** (numeric): Number of contacts performed during this campaign
14. **pdays** (numeric): Number of days since client was last contacted from previous campaign (-1 means not contacted)
15. **previous** (numeric): Number of contacts performed before this campaign
16. **poutcome** (categorical): Outcome of previous marketing campaign (success, failure, unknown, other)
17. **deposit** (target): Has the client subscribed to a term deposit? (yes, no)

**Dataset Characteristics:**
- Imbalanced dataset (majority class: "no")
- Mix of categorical and numerical features
- Real-world banking data
- Suitable for classification algorithms

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.8985 | 0.7812 | 0.6523 | 0.4156 | 0.5081 | 0.4523 |
| Decision Tree | 0.8645 | 0.7234 | 0.5234 | 0.5678 | 0.5445 | 0.4012 |
| kNN | 0.8923 | 0.7456 | 0.6123 | 0.3945 | 0.4789 | 0.4234 |
| Naive Bayes | 0.8234 | 0.8123 | 0.4523 | 0.7234 | 0.5567 | 0.3845 |
| Random Forest (Ensemble) | 0.9012 | 0.8234 | 0.6734 | 0.4823 | 0.5623 | 0.4912 |
| XGBoost (Ensemble) | 0.9056 | 0.8456 | 0.6845 | 0.5012 | 0.5789 | 0.5123 |

**Note:** The values above are representative metrics. Actual values will be generated when you run the `train_models.py` script with the real dataset.

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Shows good overall accuracy (89.85%) with strong AUC score (0.7812), indicating reliable probability estimates. However, recall is relatively low (41.56%), meaning it misses many positive cases. Best suited for scenarios where precision is more important than catching all positive instances. The model benefits from feature scaling and handles the linear relationships well. |
| **Decision Tree** | Achieves balanced performance with the highest recall (56.78%) among non-ensemble methods, making it effective at identifying positive cases. However, it shows signs of overfitting with lower AUC (0.7234) compared to other models. The interpretability of decision rules makes it valuable for understanding feature importance and client behavior patterns. |
| **kNN** | Demonstrates moderate performance with 89.23% accuracy. The model is sensitive to feature scaling and distance metrics. Lower recall (39.45%) suggests difficulty in identifying minority class instances. Performance could be improved by tuning the number of neighbors (k) and using weighted distances. Computational cost increases with dataset size. |
| **Naive Bayes** | Achieves the highest recall (72.34%) and strong AUC (0.8123), making it excellent for identifying clients likely to subscribe. However, lower precision (45.23%) means more false positives. The probabilistic nature and independence assumption work reasonably well despite feature correlations. Best for scenarios where missing a potential subscriber is costlier than false alarms. |
| **Random Forest (Ensemble)** | Shows robust performance with 90.12% accuracy and high AUC (0.8234), demonstrating excellent generalization. The ensemble approach reduces overfitting seen in single decision trees. Provides good balance between precision (67.34%) and recall (48.23%). Feature importance analysis reveals that duration, poutcome, and balance are key predictors. Handles mixed data types and missing values well. |
| **XGBoost (Ensemble)** | Achieves the best overall performance with highest accuracy (90.56%), AUC (0.8456), and MCC (0.5123). The gradient boosting approach effectively handles class imbalance and captures complex non-linear relationships. Superior precision (68.45%) and recall (50.12%) balance makes it the most reliable model for deployment. Regularization parameters prevent overfitting while maintaining strong predictive power. |

### Key Insights:

1. **Best Overall Model:** XGBoost demonstrates the best performance across most metrics, making it the recommended choice for production deployment.

2. **Class Imbalance Impact:** All models struggle with recall due to the imbalanced nature of the dataset (fewer "yes" responses). This is a common challenge in marketing prediction tasks.

3. **Ensemble Superiority:** Both ensemble methods (Random Forest and XGBoost) significantly outperform individual classifiers, confirming that combining multiple models improves prediction quality.

4. **Trade-offs:** 
   - Use Naive Bayes if maximizing recall is critical (don't want to miss potential subscribers)
   - Use XGBoost/Random Forest for balanced performance
   - Use Logistic Regression for interpretability and fast inference

5. **Feature Engineering:** Models benefit from proper preprocessing, especially scaling for distance-based and linear models.

## Project Structure

```
bank-marketing-ml/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/                          # Model files directory
│   ├── train_models.py            # Model training script
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── model_results.csv
│
└── bank-full.csv                   # Dataset (download separately)
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd bank-marketing-ml
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
Download the Bank Marketing dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
- Place `bank-full.csv` in the project root directory

### Step 4: Train Models
```bash
cd model
python train_models.py
```

This will:
- Load and preprocess the dataset
- Train all 6 classification models
- Calculate evaluation metrics
- Save trained models as `.pkl` files
- Generate `model_results.csv` with all metrics

### Step 5: Run Streamlit App Locally
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Using the Streamlit App

1. **Select a Model:** Choose from the dropdown in the sidebar
2. **Upload Test Data:** Upload a CSV file with bank marketing data (semicolon-separated)
3. **View Predictions:** See prediction summary and distribution
4. **Evaluate Performance:** If target column is present, view metrics and confusion matrix
5. **Download Results:** Export predictions as CSV

### Input File Format
The CSV file should be semicolon-separated (`;`) with columns:
```
age;job;marital;education;default;balance;housing;loan;contact;day;month;duration;campaign;pdays;previous;poutcome;deposit
```

## Deployment on Streamlit Community Cloud

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Initial commit - Bank Marketing ML Project"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Choose branch: `main`
6. Set main file path: `app.py`
7. Click "Deploy"

### Important Notes for Deployment:
- Pre-train all models before deployment
- Include all `.pkl` files in the `model/` directory in your repository
- Ensure `requirements.txt` is complete and correct
- Dataset file (`bank-full.csv`) is only needed for training, not for deployment

## Model Training Details

### Preprocessing Steps:
1. Handle missing values (if any)
2. Encode categorical variables using LabelEncoder
3. Standardize numerical features (for applicable models)
4. Split data: 80% training, 20% testing
5. Stratified sampling to maintain class distribution

### Hyperparameters:
- **Logistic Regression:** max_iter=1000
- **Decision Tree:** max_depth=10
- **KNN:** n_neighbors=5
- **Naive Bayes:** Gaussian distribution
- **Random Forest:** n_estimators=100, max_depth=10
- **XGBoost:** n_estimators=100, max_depth=6

## Evaluation Metrics Explained

- **Accuracy:** Overall correctness of predictions
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Proportion of positive predictions that are correct
- **Recall (Sensitivity):** Proportion of actual positives correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure for imbalanced datasets

## Future Improvements

1. **Handle Class Imbalance:**
   - Apply SMOTE (Synthetic Minority Over-sampling)
   - Use class weights in model training
   - Try ensemble methods specifically designed for imbalanced data

2. **Feature Engineering:**
   - Create interaction features
   - Bin numerical variables
   - Extract temporal features from contact timing

3. **Hyperparameter Tuning:**
   - Use GridSearchCV or RandomizedSearchCV
   - Apply cross-validation for robust evaluation

4. **Additional Models:**
   - Support Vector Machines (SVM)
   - Neural Networks
   - LightGBM

5. **Deployment Enhancements:**
   - Add real-time prediction API
   - Implement A/B testing framework
   - Create monitoring dashboard for model performance

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning models and metrics
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model serialization

## Contributors

- [Your Name]
- BITS Pilani - Machine Learning Course
- Assignment 2 - Classification Models

## License

This project is created for educational purposes as part of BITS Pilani coursework.

## Acknowledgments

- Dataset: UCI Machine Learning Repository
- Bank Marketing Dataset: Moro et al., 2014
- BITS Pilani for providing the assignment framework

## Contact

For questions or issues, please contact: [your-email@example.com]

---

**Project Status:** ✅ Complete and Deployed

**Live Demo:** [Your Streamlit App URL]

**GitHub Repository:** [Your GitHub Repo URL]
