# ==============================================================================
# PROJECT: QUANTIFYING THE ACCURACY-EXPLAINABILITY TRADE-OFF
# ANALYTICAL PIPELINE: PREDICTION, STABILITY, AND STRUCTURAL AUDITING
# ==============================================================================

# 1. Load Specialized Computational Libraries
library(tidyverse)  # Data wrangling 
library(caret)      # Stratified cross-validation 
library(xgboost)    # Gradient Boosting ensembles 
library(pROC)       # AUC calculation and DeLong testing 
library(lmtest)     # Ramsey RESET Test for non-linearity
library(kernelshap) # SHAP value computation 
library(shapviz)    # Clinical visualizations 
library(parallel)    # CPU optimization
library(doParallel)  # Multi-core handling
library(foreach)     # Parallel iteration
library(brglm2)      # Bias-reduction for separation cases

# 2. Data Acquisition and Preprocessing
df <- read.csv("your_dataset.csv")

# Auto-detect Cohort and Handle Target/Feature Logic
if("diagnosis" %in% colnames(df)) {
  dataset_label <- "BREAST CANCER"
  df$target <- factor(ifelse(df$diagnosis == "M", 1, 0), levels = c("0", "1"))
  feature_cols <- select(df, -diagnosis, -target, -any_of("id"))
} else if ("death_event" %in% colnames(df)) {
  dataset_label <- "HEART FAILURE"
  df$target <- factor(df$death_event, levels = c("0", "1"))
  feature_cols <- select(df, -death_event, -target, -any_of("time")) # Remove Leakage
} else if ("Outcome" %in% colnames(df)) {
  dataset_label <- "DIABETES"
  df$target <- factor(df$Outcome, levels = c("0", "1"))
  feature_cols <- select(df, -Outcome, -target)
}

# 3. Structural Validation (Ramsey RESET Test)
# Evaluates if the linear baseline misses non-linear 'geometric cliffs'
lr_full <- glm(target ~ ., data = cbind(feature_cols, target = df$target), family = "binomial")
reset_val <- resettest(lr_full, power = 2:3, type = "fitted")
print(paste(dataset_label, "Structural Misspecification p-value:", reset_val$p.value))

# 4. Experimental Setup: 10-Fold Stratified Cross-Validation
set.seed(42)
folds <- createFolds(df$target, k = 10, list = TRUE)
lr_oof_preds <- numeric(nrow(df))
xgb_oof_preds <- numeric(nrow(df))

for(i in 1:10) {
  train_idx <- setdiff(1:nrow(df), folds[[i]])
  test_idx <- folds[[i]]
  
  # Multiple Logistic Regression (MLR) - Parametric Baseline
  train_df <- cbind(feature_cols[train_idx,], target = df$target[train_idx])
  lr_mod <- glm(target ~ ., data = train_df, family = "binomial")
  lr_oof_preds[test_idx] <- predict(lr_mod, feature_cols[test_idx,], type = "response")
  
  # XGBoost - Advanced Ensemble Paradigm
  dtrain <- xgb.DMatrix(data = as.matrix(feature_cols[train_idx,]), label = as.numeric(as.character(df$target[train_idx])))
  dtest <- xgb.DMatrix(data = as.matrix(feature_cols[test_idx,]))
  xgb_mod <- xgboost(data = dtrain, nrounds = 50, objective = "binary:logistic", verbose = 0)
  xgb_oof_preds[test_idx] <- predict(xgb_mod, dtest)
}

# 5. Explanatory Stability Analysis (S)
# Measuring consistency of SHAP importance over 50 folds (Repeated 10-Fold CV)
num_repeats <- 5; k_top <- 5; all_fold_features <- list(); fold_counter <- 1

for(r in 1:num_repeats) {
  set.seed(42 + r)
  stab_folds <- createFolds(df$target, k = 10, list = TRUE)
  for(f in 1:10) {
    idx <- setdiff(1:nrow(df), stab_folds[[f]])
    d_stab <- xgb.DMatrix(data = as.matrix(feature_cols[idx,]), 
                          label = as.numeric(as.character(df$target[idx])))
    m_stab <- xgboost(data = d_stab, nrounds = 50, objective = "binary:logistic", verbose = 0)
    imp <- xgb.importance(model = m_stab)
    all_fold_features[[fold_counter]] <- head(imp$Feature, k_top)
    fold_counter <- fold_counter + 1
  }
}

# Calculate final Stability Score (S) based on feature overlap
stability_score <- sum(table(unlist(all_fold_features))[1:k_top]) / (50 * k_top)
print(paste(dataset_label, "Stability Score (S):", round(stability_score, 4)))

# 6. Performance Audit and Statistical Significance
roc_lr <- roc(df$target, lr_oof_preds, quiet = TRUE)
roc_xgb <- roc(df$target, xgb_oof_preds, quiet = TRUE)
print(roc.test(roc_lr, roc_xgb, method = "delong"))

# 7. Final SHAP Visualization (Beeswarm Impact Analysis)
full_dtrain <- xgb.DMatrix(data = as.matrix(feature_cols), label = as.numeric(as.character(df$target)))
final_xgb <- xgboost(data = full_dtrain, nrounds = 50, objective = "binary:logistic", verbose = 0)
s_values <- kernelshap(final_xgb, as.matrix(feature_cols), bg_X = as.matrix(feature_cols[sample(1:nrow(df), 100),]))
sv <- shapviz(s_values)
sv_importance(sv, kind = "beeswarm")

# ==============================================================================
# AUDIT PIPELINE: PARALLEL LOOCV & BOOTSTRAP STABILITY AUDIT
# ==============================================================================
cl <- makeCluster(4)
registerDoParallel(cl)

n <- nrow(df)
loocv_results <- foreach(i = 1:n, .combine = 'c', .packages = 'xgboost') %dopar% {
  train <- df[-i, ]
  test  <- df[i, ]
  
  dtrain <- xgb.DMatrix(data = as.matrix(train[, colnames(feature_cols)]), label = as.numeric(as.character(train$target)))
  dtest  <- xgb.DMatrix(data = as.matrix(test[, colnames(feature_cols)]))
  
  model <- xgb.train(params = list(max_depth = 6, eta = 0.3, objective = "binary:logistic"), data = dtrain, nrounds = 100, verbose = 0)
  return(predict(model, dtest))
}

stopCluster(cl)

# Bootstrap Sensitivity Loop (2,000 Iterations)
set.seed(96975)
num_iterations <- 2000
boot_auc_diffs <- numeric(num_iterations)

for(j in 1:num_iterations) {
  boot_idx <- sample(1:nrow(df), replace = TRUE)
  boot_roc_mlr <- roc(df$target[boot_idx], lr_oof_preds[boot_idx], quiet = TRUE)
  boot_auc_mlr <- as.numeric(auc(boot_roc_mlr))
  boot_roc_xgb <- roc(df$target[boot_idx], xgb_oof_preds[boot_idx], quiet = TRUE)
  boot_auc_xgb <- as.numeric(auc(boot_roc_xgb))
  boot_auc_diffs[j] <- boot_auc_xgb - boot_auc_mlr
}

print(paste("Mean AUC Difference:", mean(boot_auc_diffs)))
print(paste("95% Confidence Interval: [", quantile(boot_auc_diffs, 0.025), ",", quantile(boot_auc_diffs, 0.975), "]"))
