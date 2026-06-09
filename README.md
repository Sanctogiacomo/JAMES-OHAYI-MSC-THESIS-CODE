# A Cross-Pathology Evaluation Pipeline for Clinical Machine Learning and XAI

This repository houses the official R-based analytical pipeline for quantifying the accuracy-explainability trade-off in clinical predictive modeling. The framework validates high-dimensional non-linear ensembles against transparent parametric baselines using multi-cohort pathology datasets.

## 🚀 Key Methodological Features

* **Structural Diagnostics**: Utilizes the **Ramsey RESET Test** (`lmtest`) as an architectural gatekeeper to mathematically determine structural dataset misspecification before model assignment.
* **Separation Stabilization**: Implements bias-reduction maximum likelihood estimations (`brglm2`) to stabilize logistic baselines against perfect clinical data separation threats.
* **Algorithmic Parsimony**: Explicitly pits optimized **XGBoost Ensembles** (`xgboost`) against classical **Multiple Logistic Regression (MLR)** paradigms.
* **Explanatory Stability Score ($S$)**: Implements a custom iterative metric mapping feature-importance footprint overlap over 50 folds (Repeated 10-Fold CV) to ensure explanation robustness.
* **High-Fidelity Post-Hoc Audit**: Deploys exact Kernel SHAP algorithms (`kernelshap`) and beeswarm clinical visualizations (`shapviz`) for biomarker identification.
* **Confounding & Leakage Controls**: Built using parallelized **Leave-One-Out Cross-Validation (LOOCV)** alongside a **2,000-iteration non-parametric Bootstrap Sensitivity execution** to compute strict 95% Confidence Intervals ($\Delta$ AUC).

## 📂 Core Scripts Structure

* `APPENDIX I`: **Core Analytical Pipeline** – Automated cohort ingestion, structural data checking, 10-fold stratified CV, Stability Score ($S$) estimation, and SHAP visualization.
* `APPENDIX II`: **Scientific Audit Pipeline** – Optimized parallel processing engine via multi-core clusters (`doParallel` / `foreach`) running full LOOCV scripts.
* `APPENDIX III`: **Model Stability Audit** – High-iteration bootstrapping algorithm for calculating the mathematical complexity dividend between architectures.

## 🛠️ Environment Stack & Package Requirements

The ecosystem relies on **R (version 4.x+)**. Ensure you have the following packages installed before runtime execution:

```R
# Run this directly in your R console
install.packages(c("tidyverse", "caret", "xgboost", "pROC", "lmtest", 
                   "kernelshap", "shapviz", "parallel", "doParallel", 
                   "foreach", "brglm2"))
```

## 📊 Dataset Ingestion Note
The data preprocessing block includes built-in auto-detection logic for standard public clinical repositories:
1. **Oncology**: Breast Cancer Wisconsin (Diagnostic) Dataset (`diagnosis` indicator matching malignant/benign profiles).
2. **Cardiology**: Heart Failure Survival Dataset (`death_event` target; automatically strips the temporal `time` vector to isolate and prevent statistical data leakage).
3. **Endocrinology**: Pima Indians Diabetes Dataset (`Outcome` classification markers).

To run locally, place your cohort file in the root working directory and modify the filename string in the script:
```R
df <- read.csv("your_dataset.csv")
```
