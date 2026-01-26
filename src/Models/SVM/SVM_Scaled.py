import os
import sys
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import optuna
import mlflow

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import mlflow_helper


current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '../../../../'))
DATA_FOLDER = os.path.join(project_root, "data", "clean", "umap")


def load_split_data(data_path):
    files = {
        "Train": "train_umap_500.parquet",
        "Valid": "valid_umap_500.parquet",
        "Test": "test_umap_500.parquet",
    }

    loaded_data = {}

    for name, filename in files.items():
        file_path = os.path.join(data_path, filename)

        try:
            df = pd.read_parquet(file_path, engine="fastparquet")
        except Exception:
            df = pd.read_parquet(file_path, engine="pyarrow")

        feature_cols = sorted([c for c in df.columns if c.startswith("umap_")])
        label_col = next(c for c in ["labels", "label", "target", "class", "is_clickbait"] if c in df.columns)

        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(int)

        loaded_data[name] = (X, y)

    return loaded_data["Train"], loaded_data["Valid"], loaded_data["Test"]


def make_objective(X_tr_sc, y_tr, X_v_sc, y_v):
    def objective(trial):
        C = trial.suggest_float("C", 1e-1, 1e2, log=True)
        tol = trial.suggest_float("tol", 1e-5, 5e-4, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        params = dict(
            kernel="rbf",
            gamma="auto",
            C=C,
            tol=tol,
            class_weight=class_weight,
            max_iter=15000,
            probability=False,
            random_state=42,
        )

        model = SVC(**params)

        t0 = time.time()
        model.fit(X_tr_sc, y_tr)
        train_time = time.time() - t0

        preds = model.predict(X_v_sc)

        metrics = {
            "val_f1": f1_score(y_v, preds),
            "val_accuracy": accuracy_score(y_v, preds),
            "training_time_sec": train_time,
        }

        params_to_log = params.copy()
        params_to_log["scaler"] = "StandardScaler"

        mlflow_helper.log_optuna_trial(
            trial=trial,
            params=params_to_log,
            metrics=metrics,
            model=model,
            run_name_prefix="SVM_Trial",
        )

        return metrics["val_f1"]

    return objective


def run_experiment():
    EXPERIMENT_NAME = "Clickbait_SVM_UMAP500_Optuna"
    mlflow_helper.setup_mlflow(EXPERIMENT_NAME)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_split_data(DATA_FOLDER)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    with mlflow.start_run(run_name="SVM_Hyperparameter_Tuning"):
        study = optuna.create_study(direction="maximize")
        objective = make_objective(X_train_sc, y_train, X_val_sc, y_val)

        study.optimize(objective, n_trials=10)

        mlflow.log_metric("best_val_f1", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    with mlflow.start_run(run_name="SVM_Champion_Model"):
        best = study.best_params.copy()

        X_full_sc = np.vstack([X_train_sc, X_val_sc])
        y_full = np.concatenate([y_train, y_val])

        model = SVC(
            kernel="rbf",
            gamma="auto",
            C=best["C"],
            tol=best["tol"],
            class_weight=best["class_weight"],
            max_iter=15000,
            probability=False,
            random_state=42,
        )

        t0 = time.time()
        model.fit(X_full_sc, y_full)
        final_train_time = time.time() - t0

        mlflow.log_metric("champion_training_time_sec", final_train_time)
        mlflow.log_params(best)

        scaler_path = os.path.join(project_root, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path)

        mlflow.sklearn.log_model(model, artifact_path="champion_model")

        mlflow_helper.evaluate_and_log_metrics(
            model,
            X_test_sc,
            y_test,
            prefix="test",
            training_time=final_train_time,
        )


if __name__ == "__main__":
    run_experiment()
