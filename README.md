# Bank Marketing Campaign Prediction -- Machine Learning Project

## About This Project

This project is part of my learning journey in Machine Learning.\
The main goal is to predict whether a customer will subscribe to a term
deposit based on data collected from bank marketing campaigns.

The bank contacted customers through phone calls, sometimes multiple
times. Using this historical data, I built machine learning models to
predict if a customer will say "yes" or "no" to subscribing.

This is a binary classification problem.

------------------------------------------------------------------------

## Dataset Used

Dataset: Bank Marketing Dataset\
Source: Kaggle(https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data) \
Total Records: 45,211\
Features: 16 input features + 1 target variable

The dataset contains information such as:

-   Age\
-   Job\
-   Marital status\
-   Education\
-   Account balance\
-   Loan information\
-   Contact details\
-   Call duration\
-   Previous campaign results\
-   Target: deposit (yes/no)

The dataset is slightly imbalanced because most customers did not
subscribe.

------------------------------------------------------------------------

## What I Did in This Project

1.  Cleaned and preprocessed the dataset\
2.  Encoded categorical features\
3.  Split data into training and testing sets\
4.  Trained 6 different classification models\
5.  Compared their performance using evaluation metrics\
6.  Deployed the best model using Streamlit

------------------------------------------------------------------------

## Models Implemented

-   Logistic Regression\
-   Decision Tree\
-   K-Nearest Neighbors (KNN)\
-   Naive Bayes\
-   Random Forest\
-   XGBoost

------------------------------------------------------------------------

## Model Performance Summary

Among all models:

-   XGBoost performed the best overall.\
-   Random Forest also gave strong results.\
-   Naive Bayes gave high recall.\
-   Logistic Regression worked well as a simple baseline model.

Ensemble methods (Random Forest and XGBoost) performed better than
individual models.

------------------------------------------------------------------------

## Evaluation Metrics Used

To compare models, I used:

-   Accuracy\
-   AUC Score\
-   Precision\
-   Recall\
-   F1 Score\
-   MCC

These metrics help understand how well the models predict both "yes" and
"no" cases.

------------------------------------------------------------------------

## Project Structure

bank-marketing-ml/

    app.py  
    requirements.txt  
    README.md  
    model/  
        train_models.py  
        saved model files (.pkl)  
        model_results.csv  
    bank-full.csv  

------------------------------------------------------------------------

## How to Run the Project

Step 1: Install Requirements\
pip install -r requirements.txt

Step 2: Train Models\
cd model\
python train_models.py

Step 3: Run the Streamlit App\
streamlit run app.py

The app will open in your browser.

------------------------------------------------------------------------

## What I Learned

-   How to handle imbalanced datasets\
-   How to compare multiple ML models\
-   Why ensemble models usually perform better\
-   Importance of evaluation metrics beyond accuracy\
-   Basics of deploying ML models using Streamlit

------------------------------------------------------------------------

## Future Improvements

-   Apply SMOTE for better handling of class imbalance\
-   Perform hyperparameter tuning\
-   Add more advanced models\
-   Improve UI of the web application

------------------------------------------------------------------------

## Technologies Used

-   Python\
-   Scikit-learn\
-   XGBoost\
-   Pandas\
-   NumPy\
-   Streamlit

------------------------------------------------------------------------
