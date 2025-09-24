## Housing Prices (R • tidymodels)

A compact, reproducible pipeline to model house prices using the Kaggle Housing Prices Dataset (https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
). The workflow covers data cleaning, EDA, feature engineering, model tuning/selection, test evaluation, and a ready-to-use scoring artifact.

### Highlights

Tech: R, tidymodels, recipes, workflows, glmnet, ranger, yardstick, rsample

Targets: log(price) for stability; predictions back-transformed to the original price scale

Features: one-hot encoding for categoricals, normalization, and interaction terms (e.g., area × bedrooms, area × bathrooms)

Models: Lasso, Ridge (glmnet), Random Forest (ranger) with 10-fold CV and RMSE/MAE/R²

Deliverables: final fitted workflow (housing_model_final.rds) and a minimal scoring example

### Data

Download Housing.csv from Kaggle
Dataset: Housing Prices Dataset by Yasser Hatab — see Kaggle page for licensing/usage.

### Setup
```
install.packages(c("tidyverse","janitor","corrplot","patchwork",
                   "rsample","tidymodels","glmnet","ranger","GGally","vip"))
```

### Results
1. EDA visuals (correlations, distributions, pair plots)
2. Tune Lasso/Ridge (lambda grid) and Random Forest (mtry/min_n) with 10-fold CV
3. Select best by RMSE, fit on train, evaluate on test (RMSE/MAE/R² on original price scale)
4. Prediction on new data
