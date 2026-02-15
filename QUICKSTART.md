# Bank Marketing ML Project - Quick Start Guide

## 🚀 Quick Setup Instructions

### Step 1: Download the Dataset
1. Go to Kaggle: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
2. Download the dataset
3. Extract and place `bank-full.csv` in the project root directory

### Step 2: Train the Models
```bash
# Navigate to model directory
cd model

# Run training script (this will take a few minutes)
python train_models.py
```

This will create all model files (.pkl) needed for the Streamlit app.

### Step 3: Test Locally
```bash
# Return to project root
cd ..

# Run Streamlit app
streamlit run app.py
```

### Step 4: Prepare for GitHub & Deployment

#### Create .gitignore file:
```
bank-full.csv
*.pyc
__pycache__/
.venv/
venv/
*.log
.DS_Store
```

#### Push to GitHub:
```bash
git init
git add .
git commit -m "Bank Marketing ML Classification Project"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 5: Deploy on Streamlit Cloud
1. Visit https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: your-repo
   - Branch: main
   - Main file: app.py
5. Click "Deploy!"

## 📋 Checklist Before Submission

- [ ] Dataset downloaded and placed in project root
- [ ] Models trained successfully (all .pkl files created in model/ directory)
- [ ] Streamlit app tested locally
- [ ] All model files committed to GitHub
- [ ] README.md complete with metrics table
- [ ] requirements.txt includes all dependencies
- [ ] GitHub repository is public
- [ ] Streamlit app deployed and accessible
- [ ] Screenshot taken on BITS Virtual Lab
- [ ] PDF prepared with all required links and README content

## 📁 Required Files Structure
```
ML_Test/
├── app.py ✓
├── requirements.txt ✓
├── README.md ✓
├── QUICKSTART.md ✓
├── model/
│   ├── train_models.py ✓
│   ├── logistic_regression_model.pkl (after training)
│   ├── decision_tree_model.pkl (after training)
│   ├── knn_model.pkl (after training)
│   ├── naive_bayes_model.pkl (after training)
│   ├── random_forest_model.pkl (after training)
│   ├── xgboost_model.pkl (after training)
│   ├── scaler.pkl (after training)
│   ├── label_encoders.pkl (after training)
│   └── model_results.csv (after training)
└── bank-full.csv (download from Kaggle)
```

## ⚠️ Important Notes

1. **Dataset File**: The `bank-full.csv` file should NOT be pushed to GitHub (too large). Only the trained models are needed for deployment.

2. **Training Must Be Done First**: Before deploying to Streamlit Cloud, ensure all models are trained and .pkl files are committed to GitHub.

3. **Test Data for Streamlit**: When using the deployed app, upload smaller test CSV files (not the full dataset) due to Streamlit's free tier limitations.

4. **BITS Lab Screenshot**: Run the training script on BITS Virtual Lab and capture a screenshot showing the output.

## 🎯 Submission PDF Contents (in order)

1. **GitHub Repository Link**: https://github.com/your-username/your-repo
2. **Live Streamlit App Link**: https://your-app.streamlit.app
3. **BITS Virtual Lab Screenshot**: [Embedded image]
4. **README.md Content**: [Full README text as shown in README.md]

## 💡 Tips

- Test the Streamlit app URL in an incognito window to ensure it's publicly accessible
- Keep commit messages clear and descriptive
- If deployment fails, check Streamlit Cloud logs for missing dependencies
- Make sure all model files are less than 100MB each (GitHub limit)

## 📞 Need Help?

If you encounter issues:
1. Check that all dependencies are in requirements.txt
2. Verify all model files exist in the model/ directory
3. Test locally before deploying
4. Check Streamlit Cloud deployment logs for errors

Good luck with your assignment! 🎓
