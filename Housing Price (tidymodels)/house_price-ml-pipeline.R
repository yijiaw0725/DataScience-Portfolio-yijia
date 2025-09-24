# -----------------------------
# 1) Library
# -----------------------------
library(janitor) # clean data
library(tidyverse)
library(corrplot)
library(patchwork) # put plot together
library(rsample)
library(tidymodels)
library(glmnet)
library(ranger)

# -----------------------------
# 2) Load Data
# -----------------------------
df <- read_csv("Housing.csv") |>
  clean_names()
# Peek
glimpse(df)

# Convert "yes"/"no" and other categoricals to factors
yn_to_factor <- function(x) factor(x, levels = c("no", "yes"))
df <- df |>
  mutate(
    across(c(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea), yn_to_factor),
    furnishingstatus = factor(furnishingstatus, levels = c("unfurnished", "semi-furnished", "furnished")),
    across(c(bedrooms, bathrooms, stories, parking), as.integer)
  )

# Quick Check
summary(df$price)
sum(complete.cases(df))  # check missings

# Transform Price
df <- df |> mutate(
  log_price = log(price)     # right-skewed target -> log-transform helps linear models
)

# -----------------------------
# 3) EDA
# -----------------------------
# Numeric Correlations
num_vars <- df |>
  select(where(is.numeric))

# Correlation Plot
corr_mat <- cor(num_vars, use = "pairwise.complete.obs")
corrplot(corr_mat, method = "color", type = "upper", tl.col = "black")

# Pairwise Plot of Key Drivers
GGally::ggpairs(
  df |> select(price, area, bedrooms, bathrooms, stories, parking)
)

# -----------------------------
# 4) Visualizations
# -----------------------------
# Price Distribution (raw vs log)
p1 <- ggplot(df, aes(price)) +
  geom_histogram(bins = 30) +
  scale_x_continuous(labels = scales::label_number(scale_cut = scales::cut_si(" "))) +
  labs(title = "Distribution of Price")

p2 <- ggplot(df, aes(log_price)) +
  geom_histogram(bins = 30) +
  labs(title = "Distribution of log(Price)")

p1 + p2

# Price vs. Area with Bedrooms as Color
ggplot(df, aes(area, price, color = factor(bedrooms))) +
  geom_point(alpha = 0.6) +
  scale_y_continuous(labels = scales::label_number(scale_cut = scales::cut_si(" "))) +
  labs(color = "Bedrooms", title = "Price vs Area")

# Boxplots by Categorical (furnishing & preferred)
p3 <- ggplot(df, aes(furnishingstatus, price)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::label_number(scale_cut = scales::cut_si(" "))) +
  labs(title = "Price by Furnishing Status")

p4 <- ggplot(df, aes(prefarea, price)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::label_number(scale_cut = scales::cut_si(" "))) +
  labs(title = "Price by Preferred Area")

p3 + p4

# -----------------------------
# 5) Train/Test split
# -----------------------------
set.seed(123)
data_split <- initial_split(df, prop = 0.8, strata = price)
train <- training(data_split)
test  <- testing(data_split)

# -----------------------------
# 6) Preprocessing
# -----------------------------
# Model log(price) with One-Hot Encode Factors and Add Interactions.
base_recipe <- recipe(log_price ~ ., data = train) |>
  step_rm(price) |>                              # remove raw price from predictors
  step_zv(all_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_normalize(all_numeric_predictors()) |>
  step_interact( ~ area:starts_with("bedrooms") + area:starts_with("bathrooms") )

# -----------------------------
# 7) Models
# -----------------------------
# (a) Regularized linear regression (glmnet)
lasso_spec <- linear_reg(penalty = tune(), mixture = 1) |> set_engine("glmnet")
ridge_spec <- linear_reg(penalty = tune(), mixture = 0) |> set_engine("glmnet")

# (b) Random Forest (ranger)
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |>
  set_engine("ranger", importance = "permutation") |>
  set_mode("regression")

# -----------------------------
# 8) Workflows
# -----------------------------
wf_lasso <- workflow() |> add_model(lasso_spec) |> add_recipe(base_recipe)
wf_ridge <- workflow() |> add_model(ridge_spec) |> add_recipe(base_recipe)
wf_rf    <- workflow() |> add_model(rf_spec)    |> add_recipe(base_recipe)

# -----------------------------
# 9) Cross-validation
# -----------------------------
set.seed(123)
folds <- vfold_cv(train, v = 10, strata = price)  # stratify by price

# Grids
grid_lambda <- grid_regular(penalty(range = c(-4, 1)), levels = 30) # 1e-4..10
grid_rf <- grid_random(
  finalize(mtry(), train |> select(-log_price)),  # pick mtry based on predictors
  min_n(range = c(2, 20)),
  size = 25
)

# Tune
set.seed(123)
res_lasso <- tune_grid(wf_lasso, resamples = folds, grid = grid_lambda,
                       metrics = metric_set(rmse, rsq_trad))  # use rsq_trad to avoid constant-estimate warnings
res_ridge <- tune_grid(wf_ridge, resamples = folds, grid = grid_lambda,
                       metrics = metric_set(rmse, rsq_trad))
res_rf    <- tune_grid(wf_rf,    resamples = folds, grid = grid_rf,
                       metrics = metric_set(rmse, rsq_trad))

# -----------------------------
# 10) Compare Models (CV)
# -----------------------------
collect_metrics(res_lasso) |> filter(.metric == "rmse") |> arrange(mean) |> head()
collect_metrics(res_ridge) |> filter(.metric == "rmse") |> arrange(mean) |> head()
collect_metrics(res_rf)    |> filter(.metric == "rmse") |> arrange(mean) |> head()

autoplot(res_lasso)
autoplot(res_ridge)
autoplot(res_rf)

# Select Best Params
best_lasso <- tune::select_best(res_lasso, metric = "rmse")
best_ridge <- tune::select_best(res_ridge, metric = "rmse")
best_rf    <- tune::select_best(res_rf,    metric = "rmse")


# Finalize Workflows
final_lasso <- finalize_workflow(wf_lasso, best_lasso)
final_ridge <- finalize_workflow(wf_ridge, best_ridge)
final_rf    <- finalize_workflow(wf_rf,    best_rf)

# -----------------------------
# 11) Fit on Training & Evaluate on Test
# -----------------------------
# compute metrics on *original* price scale
# exponentiate predictions because model predicts log(price)
compute_metrics <- function(fit_wf, test_df) {
  preds <- predict(fit_wf, new_data = test_df) |>
    bind_cols(test_df |> select(price)) |>
    mutate(.pred_price = exp(.pred))
  
  yardstick::metric_set(rmse, mae, rsq)(
    data = preds,
    truth = price,
    estimate = .pred_price
  )
}

fit_lasso <- fit(final_lasso, data = train)
fit_ridge <- fit(final_ridge, data = train)
fit_rf    <- fit(final_rf,    data = train)

met_lasso <- compute_metrics(fit_lasso, test)
met_ridge <- compute_metrics(fit_ridge, test)
met_rf    <- compute_metrics(fit_rf,    test)

bind_rows(
  met_lasso |> mutate(model = "lasso"),
  met_ridge |> mutate(model = "ridge"),
  met_rf    |> mutate(model = "random_forest")
) |>
  select(model, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate) |>
  arrange(rmse)

# -----------------------------
# 12) Diagnostics & Importance
# -----------------------------
# build a combined metrics table and sort safely using
# data pronoun (avoids conflict with yardstick::rmse)
metrics_tbl <- bind_rows(
  met_lasso |> mutate(model = "lasso"),
  met_ridge |> mutate(model = "ridge"),
  met_rf    |> mutate(model = "random_forest")
) |>
  select(model, .metric, .estimate) |>
  tidyr::pivot_wider(names_from = .metric, values_from = .estimate)

# View Models Ranked by RMSE (lower better)
metrics_tbl |> arrange(.data$rmse)

# Pick the Best Model Name by RMSE using the Same Safe Arrange()
best_model_name <- metrics_tbl |>
  arrange(.data$rmse) |>
  slice(1) |>
  dplyr::pull(model) # it is ridge

# Get the Fitted Object that Corresponds to the Best Model
best_fit <- switch(best_model_name,
                   "lasso" = fit_lasso,
                   "ridge" = fit_ridge,
                   "random_forest" = fit_rf)

# Residuals vs Fitted (on the model's log scale)
test_preds <- predict(best_fit, new_data = test) |>
  bind_cols(test |> select(log_price, price, area, bedrooms, bathrooms, stories, parking))

ggplot(test_preds, aes(.pred, .pred - log_price)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Fitted (log price)", y = "Residual (log scale)",
       title = paste("Residuals vs Fitted -", best_model_name))

# Variable Importance (for random forest), not used in this case
if (best_model_name == "random_forest") {
  best_fit %>%
    extract_fit_parsnip() %>%
    vip::vip(num_features = 15)
}

# Coefficients Table for Linear Models (lasso/ridge)
if (best_model_name %in% c("lasso", "ridge")) {
  broom::tidy(best_fit) |>
    arrange(desc(abs(estimate))) |>
    slice(1:20)
}


# -----------------------------
# 13) Refit best on full training & test evaluation already done
#     (refit on all data for deployment)
# -----------------------------
final_recipe <- base_recipe
final_spec <- switch(best_model_name,
                     "lasso" = lasso_spec |> finalize_model(best_lasso),
                     "ridge" = ridge_spec |> finalize_model(best_ridge),
                     "random_forest" = rf_spec |> finalize_model(best_rf))

final_wf <- workflow() |> add_recipe(final_recipe) |> add_model(final_spec)

final_all_fit <- final_wf |> fit(df)

# -----------------------------
# 14) Save Model & Predict on New Data
# -----------------------------
saveRDS(final_all_fit, file = "housing_model_final.rds")

# Example scoring (example for test, update values to real case)
new_data <- tibble(
  price = NA,
  area = 7500,
  bedrooms = 3L,
  bathrooms = 2L,
  stories = 2L,
  mainroad = factor("yes", levels = c("no", "yes")),
  guestroom = factor("no", levels = c("no", "yes")),
  basement = factor("yes", levels = c("no", "yes")),
  hotwaterheating = factor("no", levels = c("no", "yes")),
  airconditioning = factor("yes", levels = c("no", "yes")),
  parking = 2L,
  prefarea = factor("yes", levels = c("no", "yes")),
  furnishingstatus = factor("semi-furnished",
                            levels = c("unfurnished", "semi-furnished", "furnished"))
)


# Predict on the log scale and convert back to original price scale
pred_log <- predict(final_all_fit, new_data = new_data)
pred_price <- exp(pred_log$.pred)

pred_price

# -----------------------------  END  ---------------------------------
