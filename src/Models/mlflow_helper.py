import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve)

import os  # <--- Σιγουρέψου ότι υπάρχει αυτό στην αρχή του αρχείου

def setup_mlflow(experiment_name):
    """
    Ορίζει το όνομα του πειράματος και αναγκάζει την αποθήκευση 
    στον κεντρικό φάκελο mlruns του project (Project Root).
    """
    # Βρίσκουμε το μονοπάτι του αρχείου mlflow_helper.py
    current_file_path = os.path.abspath(__file__)
    
    # Πηγαίνουμε 3 φακέλους πίσω για να βρούμε το root του project
    # (από src/Models/mlflow_helper.py -> src/Models -> src -> Project Root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    
    # Ορίζουμε τον φάκελο mlruns στο root
    mlruns_path = os.path.join(project_root, "mlruns")
    
    # Ρυθμίζουμε το MLflow να κοιτάει ΠΑΝΤΑ εκεί
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    
    mlflow.set_experiment(experiment_name)
    print(f"🚀 MLflow tracking URI set to: {mlruns_path}")
    print(f"🚀 MLflow experiment set to: {experiment_name}")

def log_optuna_trial(trial, params, metrics, model, model_name_artifact):
    """
    Καταγράφει τα αποτελέσματα ενός trial του Optuna στο MLflow.
    Δημιουργεί ένα nested run για κάθε δοκιμή.
    """
    with mlflow.start_run(nested=True):
        # 1. Καταγραφή των παραμέτρων που διάλεξε το Optuna
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        # 2. Καταγραφή των Metrics (F1, Accuracy κλπ)
        # Αν το metrics είναι λεξικό (dictionary), τα καταγράφουμε όλα
        if isinstance(metrics, dict):
            mlflow.log_metrics(metrics)
        else:
            # Αν μας ήρθε σκέτο νούμερο (π.χ. f1 score), το καταγράφουμε ως score
            mlflow.log_metric("score", metrics)

        # 3. Καταγραφή του Μοντέλου
        try:
            mlflow.sklearn.log_model(model, model_name_artifact)
        except Exception as e:
            print(f"⚠️ Δεν ήταν δυνατή η αποθήκευση του μοντέλου: {e}")
            
def evaluate_and_log_metrics(model, X_test, y_test, prefix="test"):
    """
    Υπολογίζει metrics, φτιάχνει γραφήματα και τα στέλνει στο MLflow.
    prefix: 'val' για validation set, 'test' για test set.
    """
    start_time = time.time()

    # 1. Προβλέψεις
    predictions = model.predict(X_test)

    # Προσπάθεια λήψης πιθανοτήτων για ROC-AUC (κάποια μοντέλα όπως SVM-linear δεν έχουν predict_proba)
    try:
        probs = model.predict_proba(X_test)[:, 1]
        has_probs = True
    except (AttributeError, NotImplementedError):
        has_probs = False
        print(f"⚠️ Warning: Model usually doesn't support probability output for this config.")

    inference_time = time.time() - start_time

    # 2. Υπολογισμός Metrics
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    # Καταγραφή νούμερων
    mlflow.log_metric(f"{prefix}_accuracy", acc)
    mlflow.log_metric(f"{prefix}_f1", f1)
    mlflow.log_metric(f"{prefix}_precision", precision)
    mlflow.log_metric(f"{prefix}_recall", recall)
    mlflow.log_metric(f"{prefix}_inference_time_sec", inference_time)

    if has_probs:
        auc = roc_auc_score(y_test, probs)
        mlflow.log_metric(f"{prefix}_roc_auc", auc)

    # 3. Δημιουργία Confusion Matrix Plot
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix ({prefix})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Αποθήκευση εικόνας και upload στο MLflow
    cm_filename = f"confusion_matrix_{prefix}.png"
    plt.savefig(cm_filename)
    mlflow.log_artifact(cm_filename)
    plt.close()

    # 4. Δημιουργία ROC Curve Plot (Αν έχουμε πιθανότητες)
    if has_probs:
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')  # Διαγώνιος
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({prefix})')
        plt.legend()

        roc_filename = f"roc_curve_{prefix}.png"
        plt.savefig(roc_filename)
        mlflow.log_artifact(roc_filename)
        plt.close()

    print(f"📊 Metrics logged for {prefix}: F1={f1:.4f}, Acc={acc:.4f}")
    return f1