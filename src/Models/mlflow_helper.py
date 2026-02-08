import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             average_precision_score, precision_recall_curve, log_loss, classification_report)

def setup_mlflow(experiment_name):
    current_file_path = os.path.abspath(__file__)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    mlruns_path = os.path.join(project_root, "mlruns")

    tracking_uri = Path(mlruns_path).resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"🚀 MLflow tracking URI set to: {tracking_uri}")
    print(f"🚀 MLflow experiment set to: {experiment_name}")

def log_optuna_trial(trial, params, metrics, model, run_name_prefix):
    """
    Καταγράφει τα αποτελέσματα ενός trial.
    """
    # Δημιουργούμε όνομα τύπου: "SGD_Trial_05"
    trial_name = f"{run_name_prefix}_{trial.number}"

    with mlflow.start_run(run_name=trial_name, nested=True):
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        if isinstance(metrics, dict):
            mlflow.log_metrics(metrics)
        else:
            mlflow.log_metric("score", metrics)

def evaluate_and_log_metrics(model, X_test, y_test, prefix="test", training_time=None):
    """
    Υπολογίζει metrics, φτιάχνει γραφήματα και τα στέλνει στο MLflow.
    """
    start_time = time.time()

    # 1. Προβλέψεις
    predictions = model.predict(X_test)

    # Προσπάθεια λήψης πιθανοτήτων
    try:
        probs = model.predict_proba(X_test)[:, 1]
        has_probs = True
    except (AttributeError, NotImplementedError):
        has_probs = False
        print(f"⚠️ Warning: Model usually doesn't support probability output for this config.")

    inference_time = time.time() - start_time

    # 2. Βασικά Metrics
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    metrics_to_log = {
        f"{prefix}_accuracy": acc,
        f"{prefix}_f1": f1,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_inference_time_sec": inference_time
    }

    if training_time is not None:
        metrics_to_log[f"{prefix}_training_time_sec"] = training_time

    # 3. Advanced Metrics (Log Loss & PR Curve)
    if has_probs:
        auc = roc_auc_score(y_test, probs)
        ll = log_loss(y_test, probs)
        avg_prec = average_precision_score(y_test, probs)

        metrics_to_log[f"{prefix}_roc_auc"] = auc
        metrics_to_log[f"{prefix}_log_loss"] = ll
        metrics_to_log[f"{prefix}_average_precision"] = avg_prec

    mlflow.log_metrics(metrics_to_log)

    # 4. Plots
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix ({prefix})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"confusion_matrix_{prefix}.png")
    mlflow.log_artifact(f"confusion_matrix_{prefix}.png")
    plt.close()

    if has_probs:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({prefix})')
        plt.legend()
        plt.savefig(f"roc_curve_{prefix}.png")
        mlflow.log_artifact(f"roc_curve_{prefix}.png")
        plt.close()

        # Precision-Recall Curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(rec_curve, prec_curve, label=f'AP = {avg_prec:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({prefix})')
        plt.legend()
        plt.savefig(f"pr_curve_{prefix}.png")
        mlflow.log_artifact(f"pr_curve_{prefix}.png")
        plt.close()

    # Feature Importance (για Tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Παίρνουμε τα top 20
        indices = np.argsort(importances)[::-1][:20]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top 20 Feature Importances ({prefix})")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [f"umap_{i}" for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f"feature_importance_{prefix}.png")
        mlflow.log_artifact(f"feature_importance_{prefix}.png")
        plt.close()

    # Classification Report (Text)
    report = classification_report(y_test, predictions)
    with open(f"classification_report_{prefix}.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact(f"classification_report_{prefix}.txt")

    print(f"📊 Metrics logged for {prefix}: F1={f1:.4f}, Acc={acc:.4f}")
    return f1
