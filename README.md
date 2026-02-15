# Bank Marketing Campaign Prediction - Machine Learning Classification

## Problem Statement
The goal of this project is to predict whether a client will subscribe to a term deposit based on data from direct marketing campaigns (phone calls) of a banking institution.  
This is a binary classification problem where we need to predict if the client will say "yes" or "no" to subscribing to a term deposit.

The prediction can help banks optimize their marketing strategies by targeting clients who are more likely to subscribe, thereby improving campaign efficiency and reducing costs.

---

## Dataset Description

- **Dataset Name:** Bank Marketing Data Set  
- **Source:** Kaggle  
- **Dataset Link:** [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)  
- **Dataset Size:** 45,211 instances  
- **Number of Features:** 16 input features + 1 target variable  
- **Type:** Multivariate, Sequential  
- **Classification Task:** Binary Classification  

### Features
- **age (numeric):** Age of the client  
- **job (categorical):** Type of job (admin., technician, services, management, retired, blue-collar, unemployed, entrepreneur, housemaid, unknown, self-employed, student)  
- **marital (categorical):** Marital status (married, single, divorced)  
- **education (categorical):** Education level (primary, secondary, tertiary, unknown)  
- **default (categorical):** Has credit in default? (yes, no)  
- **balance (numeric):** Average yearly balance in euros  
- **housing (categorical):** Has housing loan? (yes, no)  
- **loan (categorical):** Has personal loan? (yes, no)  
- **contact (categorical):** Contact communication type (cellular, telephone, unknown)  
- **day (numeric):** Last contact day of the month  
- **month (categorical):** Last contact month of year (jan, feb, mar, ..., nov, dec)  
- **duration (numeric):** Last contact duration in seconds  
- **campaign (numeric):** Number of contacts performed during this campaign  
- **pdays (numeric):** Number of days since client was last contacted from previous campaign (-1 means not contacted)  
- **previous (numeric):** Number of contacts performed before this campaign  
- **poutcome (categorical):** Outcome of previous marketing campaign (success, failure, unknown, other)  
- **deposit (target):** Has the client subscribed to a term deposit? (yes, no)  

**Dataset Characteristics:**
- Imbalanced dataset (~88% "no", ~12% "yes")  
- Mix of 7 categorical and 10 numerical features  
- Minimal missing values, handled through preprocessing  
- Real-world banking campaign data  
- Includes time-based features (day, month, duration)  

---

## Models Used & Evaluation Metrics

| ML Model            | Accuracy | AUC    | Precision | Recall  | F1 Score | MCC   |
|---------------------|----------|--------|-----------|---------|----------|-------|
| Logistic Regression | 0.7971   | 0.8729 | 0.7957    | 0.7694  | 0.7823   | 0.5928|
| Decision Tree       | 0.8200   | 0.8470 | 0.7987    | 0.8289  | 0.8135   | 0.6400|
| kNN                 | 0.7900   | 0.8520 | 0.7966    | 0.7476  | 0.7713   | 0.5785|
| Naive Bayes         | 0.7546   | 0.8088 | 0.7150    | 0.8015  | 0.7558   | 0.5141|
| Random Forest       | 0.8401   | 0.9127 | 0.8138    | 0.8592  | 0.8359   | 0.6812|
| XGBoost             | 0.8522   | 0.9221 | 0.8285    | 0.8677  | 0.8476   | 0.7050|

---

## Key Insights & Conclusions
- **Best Overall Model:** XGBoost (Accuracy 85.22%, AUC 0.9221, F1 0.8476, MCC 0.7050)  
- **Ensemble Methods Dominance:** Random Forest and XGBoost outperform individual classifiers.  
- **Class Imbalance Handling:** Ensemble methods achieve recall >85%, effectively identifying minority class.  
- **AUC as Key Metric:** All models show good discriminative ability (0.80–0.92).  
- **Trade-offs:**
  - Maximum Accuracy → XGBoost  
  - Interpretability → Decision Tree / Logistic Regression  
  - Speed → Logistic Regression / Naive Bayes  
  - Balanced Performance → Random Forest / XGBoost  

---

## Project Structure
- **app.py** → Streamlit web application  
- **requirements.txt** → Dependencies  
- **README.md** → Documentation  
- **bank-full.csv** → Dataset  
- **model/** → Trained models & scripts  
  - `train_models.py` → Preprocessing & training  
  - `model_results.csv` → Evaluation metrics  
  - `.pkl` files → Saved models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost, Scaler)  

---

## Technologies Used
- Python 3.12.7  
- Scikit-learn  
- XGBoost  
- Pandas, NumPy  
- Streamlit  
- Matplotlib, Seaborn  
- Joblib  

---

## How to Run the Project
1. Clone the repository  
2. Install dependencies  
3. Train the model  
4. Run the Streamlit app locally  
5. Access the application  

**Deployment:**  
- Platform: Streamlit Community Cloud  
- Live App: [Streamlit App Link](https://ml-bank-marketing-campaign-prediction-k2n5ywcqhfunvaujnqexke.streamlit.app/)  
- GitHub Repo: [ML_Assignment_2_Banking](https://github.com/prakhar8299/ML_-Bank-Marketing-Campaign-Prediction.git)  

---

## Model Training Details
- **Preprocessing:** Missing values removed, LabelEncoder for categorical, StandardScaler for scaling  
- **Train-Test Split:** 80-20 with stratification  

**Configurations:**
- Logistic Regression → max_iter=1000, solver='lbfgs'  
- Decision Tree → max_depth=10, random_state=42  
- kNN → n_neighbors=5, metric='euclidean'  
- Naive Bayes → Gaussian assumption  
- Random Forest → n_estimators=100, max_depth=10, random_state=42  
- XGBoost → n_estimators=100, max_depth=6, learning_rate=0.3  

---

## Evaluation Metrics Explanation
- **Accuracy:** Overall correctness  
- **AUC:** Ability to distinguish classes  
- **Precision:** Correct positive predictions  
- **Recall:** Correctly identified positives  
- **F1 Score:** Harmonic mean of precision & recall  
- **MCC:** Correlation between predictions & actual values  

---

## Future Enhancements
- Implement SMOTE for imbalance  
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)  
- Feature importance visualization  
- Real-time single-record predictions  
- Model explainability (SHAP, LIME)  
- A/B testing for performance comparison  
- Database integration for predictions & feedback  

---

## Author
**Prakhar Singh**  
Created as part of Machine Learning Assignment 2 - BITS Pilani
