# Data Science Portfolio – Yijia

This is my personal data science portfolio.  
This repository showcases selected projects I’ve completed, demonstrating skills in **data analysis, machine learning, and business problem-solving**.  
All datasets are either public, no proprietary or confidential data is included.

## Projects

### 1. [Bank Marketing Campaign Optimization](./BankCall%20Optimizer)
**Goal:** Predict whether a bank customer will subscribe to a term deposit product based on historical telemarketing data.  
**Highlights:**  
- Performed **EDA** to uncover patterns in customer demographics and past interactions.  
- Built **XGBoost** and **GBDT** models, improving prediction accuracy over baseline models.  
- Identified top features influencing subscription decisions, helping optimize marketing strategy.  

**Tech:** Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

---

### 2. [Data Science Job Market Analysis](./DS-Job-Market)
**Goal:** Analyze data science job postings to uncover salary trends, required skills, and location preferences.  
**Highlights:**  
- Cleaned and standardized raw job posting data.  
- Created salary and demand visualizations to support career planning.  

**Tech:** Python, pandas, matplotlib, seaborn

---

### 3. [Titanic Survival Prediction](./Titanic)
**Goal:** Predict passenger survival on the Titanic using demographic and travel data.  
**Highlights:**  
- Engineered new features (family size, title extraction) to improve model performance.  
- Compared Logistic Regression, Random Forest, and Gradient Boosting models.  
- Achieved competitive accuracy on Kaggle leaderboard.  

**Tech:** Python, pandas, scikit-learn, seaborn

### 4. [Housing Price Prediction](./Housing%20Price (tidymodels))
**Goal:** Predict home sale prices using the Kaggle Housing Prices dataset.

**Highlights:**  
- Cleaned and standardized data; converted categorical “yes/no” fields to factors.
- Modeled log(price) for stability; back-transformed predictions to the original scale.
- Built a tidymodels recipe with one-hot encoding, normalization, and interactions (area × bedrooms, area × bathrooms).
- Compared Lasso, Ridge, and Random Forest using 10-fold CV; selected the best by RMSE (Ridge in this run).
- Reported RMSE/MAE/R² on the original price scale; saved a deployable workflow (housing_model_final.rds) and included a minimal scoring example.

 **Tech:** R, tidymodels (recipes, workflows), glmnet, ranger, yardstick, rsample, ggplot2.

---

## Skills Demonstrated
- **Programming & Data Handling:** Python, R, pandas, NumPy  
- **Machine Learning:** scikit-learn, XGBoost, Random Forest, Gradient Boosting  
- **Data Visualization:** matplotlib, seaborn, plotly  


---

## Contact
- **Email:** [yijiaw0725@gmail.com]
