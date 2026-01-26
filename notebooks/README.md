# Clickbait Detection using Gemma LLM & UMAP

This repository contains the end-to-end machine learning workflow for my thesis on **Clickbait Detection**. The project utilizes state-of-the-art **Google Gemma LLM** embeddings, dimensionality reduction via **UMAP**, and a rigorous comparative analysis of various classifiers.

A key scientific finding of this work is the **"Scaling Paradox"**, where distance-based models performed significantly better on *raw* UMAP embeddings compared to standard-scaled ones.

---

##  Project Pipeline

The notebooks are numbered to reflect the execution order.

### 1. Data Preparation & Feature Engineering
The foundational steps to process text and generate semantic features.

- **`01_data_overview.ipynb`**: Exploratory Data Analysis (EDA), sanity checks on raw sources, and label balance inspection.
- **`02_clean_merge.ipynb`**: Text cleaning, normalization, and merging distinct dataset sources into a unified corpus.
- **`03_feature_engineering.ipynb`**: **(Critical Step)**
  - Generating high-dimensional embeddings using **Google's Gemma LLM**.
  - Applying **UMAP** to reduce dimensions to 500 components while preserving topological structure.
  - Splitting data into Train/Validation/Test sets (saved as Parquet).

### 2. Preliminary Modeling
- **`04_modeling_embeddings.ipynb`**: Initial experimental runs to gauge general model performance on embeddings.
- **`05_evaluation_report.ipynb`**: Early evaluation metrics and error analysis scripts.

---

##  The Experiments: Scaling & Architecture Study

This section constitutes the core research. We investigate how different algorithm families (Trees vs. Linear vs. Kernels) interact with the geometry of UMAP embeddings. All experiments use **Optuna** for hyperparameter tuning and **MLflow** for tracking.

### Best Model
- **`06_Gradient_Booster.ipynb`**: 
  - Implementation of **Gradient Boosting Classifier**.
  - **Result:** Top performance (~91% Accuracy, ~0.88 F1).
  - Demonstrates robustness to unscaled features and non-linear decision boundaries.

###  The Scaling Study (Linear Models)
Investigating the impact of `StandardScaler` on UMAP manifolds.

- **`09_Log_Reg_No_Scaling.ipynb`**: 
  - Logistic Regression on **raw** UMAP data.
  - **Result:** Excellent performance (~0.86 F1), proving UMAP successfully "unrolled" the Gemma manifold into a linearly separable space.
- **`08_Log_Reg_Scaled.ipynb`**: 
  - Logistic Regression with **StandardScaling**.
  - **Observation:** Performance drop (~0.70 F1) due to distortion of density-based information encoded by UMAP.
- **`07_SGD.ipynb`**: 
  - Stochastic Gradient Descent baseline.

###  The Scaling Study (SVM)
- **`11_SVM_No_Scaling.ipynb`**: 
  - SVM (RBF Kernel) on **raw** UMAP data.
- **`10_SVM_Scaled.ipynb`**: 
  - SVM with **StandardScaling**.
  - **Observation:** Failed to converge efficiently and showed degraded performance compared to the unscaled version.

---

##  Final Conclusions

- **`Results.ipynb`**: 
  - **The Final Report.** Consolidated analysis of all 11 notebooks.
  - Contains comparative visualizations (Bar charts, Leaderboards).
  - Scientific discussion on why **Gradient Boosting** won and why **Scaling** harmed the topological features of Gemma+UMAP.

---

## ️ Tech Stack

- **LLM / Embeddings**: Google Gemma
- **Dimensionality Reduction**: UMAP
- **Orchestration**: Python, Jupyter
- **Tracking**: MLflow
- **Optimization**: Optuna
- **Data Management**: Git LFS (for large .parquet files)

## How to Reproduce

1. **Install Dependencies**: Ensure `umap-learn`, `optuna`, `mlflow`, `scikit-learn`, `xgboost` are installed.
2. **Data**: Place raw data in `data/raw/` (or ensure `data/clean/umap/` contains the parquet files).
3. **Run Pipeline**: Execute notebooks `01` through `03` to generate features.
4. **Run Experiments**: Execute notebooks `06` through `11` to train models and log artifacts.
5. **View Results**: Open `Results.ipynb` for the summary.