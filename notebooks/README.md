# 📂 Notebook Suite: Clickbait Detection with UMAP & ML

This folder contains the complete end-to-end workflow for the **Clickbait Detection Thesis**. The pipeline ranges from raw data preprocessing and dimensionality reduction using **UMAP** on **Gemma LLM Embeddings** to advanced model training, hyperparameter tuning via **Optuna**, and experiment tracking with **MLflow**.

## 🚀 Main Pipeline (Data Prep)

The foundational steps to prepare the text data and generate embeddings.

- **`01_data_overview.ipynb`**: Sanity checks on raw sources, exploratory data analysis (EDA), and label balance inspection.
- **`02_clean_merge.ipynb`**: Text cleaning, normalization, and merging across different dataset sources.
- **`03_feature_engineering_umap.ipynb`**: 
  - Generation of embeddings using **Google's Gemma LLM**.
  - Dimensionality reduction using **UMAP** (500 components).
  - Splitting into Train/Validation/Test sets (saved as Parquet).

---

## 🧪 Modeling Experiments (The Core Analysis)

This section contains the training scripts converted into interactive notebooks. Each notebook performs Hyperparameter Tuning (Optuna) and tracks metrics via MLflow.

### 🏆 Champion Model
- **`06_Gradient_Booster.ipynb`**: 
  - Implementation of the **Gradient Boosting Classifier**.
  - **Result:** Best performing model (~91% Accuracy, ~0.88 F1).
  - Demonstrates robustness to unscaled UMAP features derived from Gemma.

### 📉 Linear & Distance-Based Models (The Scaling Study)
A critical part of this thesis was investigating the "Scaling Paradox" on UMAP embeddings.

- **`05_Logistic_Regression_NoScaling.ipynb`**: 
  - Training on **raw UMAP embeddings**.
  - **Key Finding:** High performance (~0.86 F1), proving UMAP's ability to "unroll" the Gemma manifold into a linearly separable space.
- **`05_Logistic_Regression_Scaled.ipynb`**: 
  - Training with `StandardScaler`.
  - **Observation:** Significant performance drop (~0.70 F1) due to topological distortion.
- **`05_SVM_NoScaling.ipynb`**: 
  - SVM with RBF Kernel on raw data.
- **`05_SVM_Scaled.ipynb`**: 
  - SVM with Standard Scaling (traditional approach), which failed to converge efficiently.

### ⚡ Baselines
- **`SGD.ipynb`**: Stochastic Gradient Descent classifier as a lightweight baseline.

---

## 📊 Evaluation & Conclusions

- **`07_Results_Discussion.ipynb`**: 
  - Consolidated analysis of all experiments.
  - Comparative visualizations (Bar charts, Leaderboards).
  - Final discussion on **Tree-based vs. Linear models** and the impact of normalization on manifold learning.

---

## 🛠️ Tech Stack & How to Use

- **Embeddings Source**: Google Gemma LLM.
- **Dimensionality Reduction**: UMAP (Uniform Manifold Approximation and Projection).
- **Experiment Tracking**: All runs are logged in `mlruns/` via **MLflow**.
- **Optimization**: Hyperparameters are tuned using **Optuna** (TPE Sampler).
- **Data Format**: Large files use `.parquet` format and are tracked via **Git LFS**.

### Instructions
1. Ensure data is located in `data/clean/umap/`.
2. Run the **Modeling** notebooks to reproduce training and artifact generation (models & scalers).
3. Open `07_Results_Discussion.ipynb` to view the final comparative report.