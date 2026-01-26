import pandas as pd
import numpy as np
import optuna
import mlflow
import sys
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# --- ΡΥΘΜΙΣΕΙΣ PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlflow_helper

# Δυναμικός εντοπισμός των αρχείων δεδομένων
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '../../..'))
DATA_FOLDER = os.path.join(project_root, 'data', 'clean', 'umap')


def load_split_data(data_path):
    """
    Φορτώνει τα Train/Valid/Test (χωρίς scaling).
    """
    files = {
        "Train": "train_umap_500.parquet",
        "Valid": "valid_umap_500.parquet",
        "Test": "test_umap_500.parquet"
    }

    loaded_data = {}
    possible_label_cols = ['labels', 'label', 'target', 'class', 'is_clickbait']

    print(f"⏳ Έναρξη φόρτωσης δεδομένων από: {data_path}")

    for name, filename in files.items():
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path):
            print(f"❌ Το αρχείο {filename} δεν βρέθηκε.")
            sys.exit(1)

        try:
            df = pd.read_parquet(file_path, engine='fastparquet')
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine='pyarrow')
            except Exception as e:
                print(f"⛔ Σφάλμα στο {filename}: {e}")
                sys.exit(1)

        feature_cols = [c for c in df.columns if c.startswith("umap_")]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c not in possible_label_cols]

        label_col = next((c for c in possible_label_cols if c in df.columns), None)

        if label_col is None:
            remaining = [c for c in df.columns if c not in feature_cols]
            if len(remaining) == 1:
                label_col = remaining[0]
            else:
                prefix = filename.split('_')[0]
                ext_path = os.path.join(data_path, f"{prefix}_labels.csv")
                if os.path.exists(ext_path):
                    df_l = pd.read_csv(ext_path)
                    y = df_l.iloc[:, 0].values.astype(int)
                    X = df[feature_cols].values.astype(np.float32)
                    loaded_data[name] = (X, y)
                    print(f"   ✅ {name} loaded: X={X.shape}, y={y.shape}")
                    continue
                else:
                    sys.exit(1)

        if label_col in feature_cols:
            feature_cols.remove(label_col)

        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(int)
        loaded_data[name] = (X, y)
        print(f"   ✅ {name} loaded: X={X.shape}, y={y.shape}")

    return loaded_data["Train"], loaded_data["Valid"], loaded_data["Test"]


def objective(trial, X_tr, y_tr, X_v, y_v):
    """
    Objective function για Logistic Regression (ΧΩΡΙΣ SCALING).
    """
    solver = "saga"
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C = trial.suggest_float("C", 1e-4, 10.0, log=True)

    params = {
        "solver": solver,
        "penalty": penalty,
        "C": C,
        "max_iter": 2000,  # Αυξημένο όριο για να βοηθήσει τη σύγκλιση χωρίς scaler
        "random_state": 42,
        "n_jobs": -1
    }

    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

    # --- NO SCALING ---
    # Χρησιμοποιούμε τα δεδομένα όπως είναι (X_tr, X_v)
    model = LogisticRegression(**params)

    start_time = time.time()
    model.fit(X_tr, y_tr)
    training_time = time.time() - start_time

    preds = model.predict(X_v)
    f1 = f1_score(y_v, preds)
    acc = accuracy_score(y_v, preds)

    metrics = {"val_f1": f1, "val_accuracy": acc, "training_time_sec": training_time}
    mlflow_helper.log_optuna_trial(trial, params, metrics, model, run_name_prefix="LR_Trial")

    return f1


def run_experiment():
    EXPERIMENT_NAME = "Clickbait_LR_UMAP_NoScaling"

    mlflow_helper.setup_mlflow(EXPERIMENT_NAME)
    print(f"\n🚀 Έναρξη Πειράματος: {EXPERIMENT_NAME}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_split_data(DATA_FOLDER)

    print("ℹ️ Σημείωση: Δεν εφαρμόζεται Scaling (No Scaling).")

    # --- PHASE 1: Tuning ---
    print("\n🔍 ΦΑΣΗ 1: Hyperparameter Tuning (Optuna)...")

    with mlflow.start_run(run_name="🔍_LR_Hyperparameter_Tuning") as tuning_run:
        mlflow.log_param("dataset", "UMAP_500")
        mlflow.log_param("scaling", "None")

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # 20 Trials
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)

        print(f"🏆 Best Params: {study.best_params}")
        print(f"🏆 Best Val F1: {study.best_value:.4f}")

    # --- PHASE 2: Champion Model ---
    print("\n👑 ΦΑΣΗ 2: Champion Model Training...")

    with mlflow.start_run(run_name="👑_LR_Champion_Model") as final_run:
        best_params = study.best_params
        best_params.update({
            "solver": "saga",
            "max_iter": 2000,
            "random_state": 42,
            "n_jobs": -1
        })
        mlflow.log_params(best_params)
        mlflow.log_param("dataset", "UMAP_500_NoScaling")

        # --- NO SCALING ---
        # Ενώνουμε τα δεδομένα χωρίς να εφαρμόσουμε scaler
        X_full = np.concatenate((X_train, X_val))
        y_full = np.concatenate((y_train, y_val))

        final_model = LogisticRegression(**best_params)

        start_t = time.time()
        final_model.fit(X_full, y_full)
        final_train_time = time.time() - start_t
        print(f"⏱️ Training Time: {final_train_time:.2f} sec")

        mlflow.sklearn.log_model(final_model, artifact_path="champion_model")

        print("📈 Αξιολόγηση στο Test Set...")
        mlflow_helper.evaluate_and_log_metrics(
            final_model,
            X_test,
            y_test,
            prefix="test",
            training_time=final_train_time
        )

        print(f"\n✅ ΤΕΛΟΣ! Run ID: {final_run.info.run_id}")


if __name__ == "__main__":
    run_experiment()