import pandas as pd
import numpy as np
import optuna
import mlflow
import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import sys
import os
import warnings

# Optional: reduce sklearn deprecation noise
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlflow_helper


current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '../../../'))
DATA_FOLDER = os.path.join(project_root, "data", "clean", "umap")


def load_split_data(data_path):
    files = {
        "Train": "train_umap_500.parquet",
        "Valid": "valid_umap_500.parquet",
        "Test": "test_umap_500.parquet"
    }

    loaded_data = {}

    for name, filename in files.items():
        file_path = os.path.join(data_path, filename)
        print("LOADING SPLIT:", name, filename)

        try:
            df = pd.read_parquet(file_path, engine="fastparquet")
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine="pyarrow")
            except Exception as e:
                raise RuntimeError(f"Failed to read parquet {filename}: {e}")

        feature_cols = sorted([c for c in df.columns if c.startswith("umap_")])

        if len(feature_cols) != 500:
            raise ValueError(f"{feature_cols}: the number of features are {len(feature_cols)} in {filename} !")

        possible_label_cols = ["labels", "label", "target", "class", "is_clickbait"]
        label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            raise ValueError(f"The {filename} file is without labels")

        y = df[label_col].values.astype(int)
        X = df[feature_cols].values.astype(np.float32)

        if len(X) != len(y):
            raise ValueError(f"features!=labels in the {filename} file")

        loaded_data[name] = (X, y)
        print(f"{name} loaded: X={X.shape}, y={y.shape}")

    print("LOADED KEYS:", loaded_data.keys())
    return loaded_data["Train"], loaded_data["Valid"], loaded_data["Test"]


def objective(trial, X_tr, y_tr, X_v, y_v):
    C = trial.suggest_float("C", 1e-4, 1e2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

    # Unified solver search space to avoid Optuna dynamic categorical error
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])

    # Reject invalid combinations
    if penalty == "elasticnet" and solver != "saga":
        raise optuna.exceptions.TrialPruned()
    if penalty == "l1" and solver == "lbfgs":
        raise optuna.exceptions.TrialPruned()
    # lbfgs supports only l2 (and none). liblinear supports l1/l2. saga supports l1/l2/elasticnet.

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)
    max_iter = trial.suggest_int("max_iter", 300, 4000)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_v_sc = scaler.transform(X_v)

    params = dict(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        tol=tol,
        max_iter=max_iter,
        random_state=42,
    )
    if l1_ratio is not None:
        params["l1_ratio"] = l1_ratio

    model = LogisticRegression(**params)

    t0 = time.time()
    model.fit(X_tr_sc, y_tr)
    train_time = time.time() - t0

    preds = model.predict(X_v_sc)
    f1 = f1_score(y_v, preds)
    acc = accuracy_score(y_v, preds)

    metrics = {
        "val_f1": f1,
        "val_accuracy": acc,
        "training_time_sec": train_time,
    }

    # Keep scaler info as param (not metric)
    params_to_log = params.copy()
    params_to_log["scaler"] = "StandardScaler"

    mlflow_helper.log_optuna_trial(
        trial=trial,
        params=params_to_log,
        metrics=metrics,
        model=model,
        run_name_prefix="LR_Trial",
    )

    return f1


def run_experiment():
    EXPERIMENT_NAME = "Clickbait_LogReg_UMAP500_Optuna_Scaled"

    mlflow_helper.setup_mlflow(EXPERIMENT_NAME)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_split_data(DATA_FOLDER)

    with mlflow.start_run(run_name="LR_Hyperparameter_Tuning"):
        mlflow.log_param("dataset", "UMAP_500")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("scaler", "StandardScaler")

        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda t: objective(t, X_train, y_train, X_val, y_val),
            n_trials=20,
            catch=(optuna.exceptions.TrialPruned,)
        )

        mlflow.log_metric("best_val_f1", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

        print("Best Val F1:", study.best_value)
        print("Best Params:", study.best_params)

    with mlflow.start_run(run_name="LR_Champion_Model"):
        best = study.best_params.copy()

        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        scaler = StandardScaler()
        X_full_sc = scaler.fit_transform(X_full)
        X_test_sc = scaler.transform(X_test)

        penalty = best["penalty"]
        solver = best["solver"]

        champion_params = dict(
            C=best["C"],
            penalty=penalty,
            solver=solver,
            class_weight=best["class_weight"],
            tol=best["tol"],
            max_iter=best["max_iter"],
            random_state=42,
        )
        if penalty == "elasticnet":
            champion_params["l1_ratio"] = best["l1_ratio"]

        model = LogisticRegression(**champion_params)

        t0 = time.time()
        model.fit(X_full_sc, y_full)
        final_train_time = time.time() - t0

        mlflow.log_params(champion_params)
        mlflow.log_param("dataset", "UMAP_500")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_metric("champion_training_time_sec", final_train_time)

        # Optional: log scaler artifact for reproducibility
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact("scaler.pkl")

        mlflow.sklearn.log_model(model, artifact_path="champion_model")

        mlflow_helper.evaluate_and_log_metrics(
            model,
            X_test_sc,
            y_test,
            prefix="test",
            training_time=final_train_time
        )

        print("Finished. Check MLflow UI for runs.")


if __name__ == "__main__":
    run_experiment()

