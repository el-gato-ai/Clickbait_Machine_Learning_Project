import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, roc_curve)


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