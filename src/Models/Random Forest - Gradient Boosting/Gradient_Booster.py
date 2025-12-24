import pandas as pd
import numpy as np
import optuna
import mlflow
import sys
import os
import time  # <--- Χρειαζόμαστε το time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- ΡΥΘΜΙΣΕΙΣ PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlflow_helper

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '../../..'))
DATA_FOLDER = os.path.join(project_root, 'data', 'clean', 'umap')


def load_split_data(data_path):
    # (Ο κώδικας φόρτωσης παραμένει ίδιος όπως τον φτιάξαμε πριν - Copy Paste από το προηγούμενο)
    files = {"Train": "train_umap_500.parquet", "Valid": "valid_umap_500.parquet", "Test": "test_umap_500.parquet"}
    loaded_data = {}
    possible_label_cols = ['labels', 'label', 'target', 'class', 'is_clickbait']

    print(f"⏳ Έναρξη φόρτωσης δεδομένων από: {data_path}")
    for name, filename in files.items():
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path): sys.exit(f"❌ Missing {filename}")

        try:
            df = pd.read_parquet(file_path, engine='fastparquet')
        except:
            df = pd.read_parquet(file_path, engine='pyarrow')

        feature_cols = [c for c in df.columns if c.startswith("umap_")]
        if not feature_cols: feature_cols = [c for c in df.columns if c not in possible_label_cols]

        label_col = next((c for c in possible_label_cols if c in df.columns), None)
        if not label_col:
            rem = [c for c in df.columns if c not in feature_cols]
            if len(rem) == 1:
                label_col = rem[0]
            else:
                sys.exit(f"❌ No label found in {filename}")

        if label_col in feature_cols: feature_cols.remove(label_col)

        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(int)
        loaded_data[name] = (X, y)
        print(f"   ✅ {name} loaded: {X.shape}")

    return loaded_data["Train"], loaded_data["Valid"], loaded_data["Test"]


def objective(trial, X_tr, y_tr, X_v, y_v):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42
    }

    model = GradientBoostingClassifier(**params)

    # --- Μέτρηση Χρόνου Εκπαίδευσης (Trial) ---
    start_time = time.time()
    model.fit(X_tr, y_tr)
    training_time = time.time() - start_time
    # -----------------------------------------------

    preds = model.predict(X_v)
    f1 = f1_score(y_v, preds)
    acc = accuracy_score(y_v, preds)

    # Περνάμε και το training_time στα metrics που καταγράφονται
    metrics = {"val_f1": f1, "val_accuracy": acc, "training_time_sec": training_time}
    mlflow_helper.log_optuna_trial(trial, params, metrics, model, "gb_trial")

    return f1


def run_experiment():
    EXPERIMENT_NAME = "Clickbait_GradientBoosting_UMAP_Final_V2"
    mlflow_helper.setup_mlflow(EXPERIMENT_NAME)
    print(f"\n🚀 Έναρξη Πειράματος: {EXPERIMENT_NAME}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_split_data(DATA_FOLDER)

    with mlflow.start_run(run_name="GB_Optimization_Full_Metrics") as run:
        mlflow.log_param("dataset", "UMAP_500")

        # Optuna
        print("\n🔎 Tuning Hyperparameters...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=15)

        print(f"🏆 Best Params: {study.best_params}")

        # Champion Model Training
        print("\n⚙️ Training Champion Model...")
        best_params = study.best_params
        best_params["random_state"] = 42
        final_model = GradientBoostingClassifier(**best_params)

        X_full = np.concatenate((X_train, X_val))
        y_full = np.concatenate((y_train, y_val))

        # --- NEW: Μέτρηση Χρόνου (Champion Model) ---
        start_t = time.time()
        final_model.fit(X_full, y_full)
        final_train_time = time.time() - start_t
        print(f"⏱️ Training Time: {final_train_time:.2f} sec")
        # --------------------------------------------

        mlflow.sklearn.log_model(final_model, artifact_path="champion_model")

        # Evaluation
        print("\n📈 Evaluating on Test Set...")
        # Περνάμε το training_time στη συνάρτηση για να καταγραφεί μαζί με τα υπόλοιπα
        mlflow_helper.evaluate_and_log_metrics(final_model, X_test, y_test, prefix="test",
                                               training_time=final_train_time)

        print(f"\n✅ Ολοκληρώθηκε! Run ID: {run.info.run_id}")


if __name__ == "__main__":
    run_experiment()