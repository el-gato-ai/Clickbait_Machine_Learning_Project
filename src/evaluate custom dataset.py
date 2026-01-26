import pandas as pd
import numpy as np
import mlflow
import sys
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# --- ΡΥΘΜΙΣΕΙΣ PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Models')))
try:
    import mlflow_helper
except ImportError:
    # Fallback αν τρέχει από άλλο φάκελο
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import mlflow_helper

# ==========================================
# ⚙️ ΡΥΘΜΙΣΕΙΣ ΧΡΗΣΤΗ
# ==========================================

# 1. Path του Gold Dataset (Test set)
CUSTOM_DATA_PATH = r"/Users/nikosgatos/PycharmProjects/Clickbait_Machine_Learning_Project/data/clean/umap/test_umap_500.parquet"

# 2. Path του TRAIN Dataset (Απαραίτητο για να φτιάξουμε τον Scaler!)
TRAIN_DATA_PATH = r"/Users/nikosgatos/PycharmProjects/Clickbait_Machine_Learning_Project/data/clean/umap/train_umap_500.parquet"

# 3. Τα Paths των .pkl αρχείων
MODELS_TO_EVALUATE = {
    # Μοντέλα που θέλουν RAW data (χωρίς Scaling)
    "SGD_Classifier": {
        "path": r"/Users/nikosgatos/PycharmProjects/Clickbait_Machine_Learning_Project/mlruns/236777006947026757/models/m-022afa5b1f9848768d11c9390253ec71/artifacts/model.pkl",
        "needs_scaling": False
    },
    "Gradient_Boosting": {
        "path": r"/Users/nikosgatos/PycharmProjects/Clickbait_Machine_Learning_Project/mlruns/176203313038895818/models/m-11223ec66e124e829ece6083c7b53cc5/artifacts/model.pkl",
        "needs_scaling": False
    },
    "SVM_NoScaling": {
        "path": r"/Users/nikosgatos/PycharmProjects/Clickbait_Machine_Learning_Project/mlruns/664524367874882829/models/m-316b355112e14df284738170b470bf27/artifacts/model.pkl",
        "needs_scaling": False
    },
    "Logistic_Regression_NoScaling": {
        "path": r"/Users/nikosgatos/PycharmProjects/Clickbait_Machine_Learning_Project/mlruns/961020367779049974/models/m-f5332dd335424d659711f5acb87d7eda/artifacts/model.pkl",
        "needs_scaling": False
    },
    "SVM_Scaled": {
        "path": r"/Users/nikosgatos/PycharmProjects/Clickbait_Machine_Learning_Project/mlruns/629225326235206482/models/m-f917c98e8f4c4e0f895ef460199ce813/artifacts/model.pkl",
        "needs_scaling": True
    },
    "Logistic_Regression_Scaled": {
        "path": r"/Users/nikosgatos/PycharmProjects/Clickbait_Machine_Learning_Project/mlruns/444470392771103284/models/m-ad71b18c165c4043b615b5fc234a675d/artifacts/model.pkl",
        "needs_scaling": True
    },
}

NEW_EXPERIMENT_NAME = "Final_Evaluation_Rescaled"


# ==========================================

def load_data(path):
    print(f"   📂 Loading: {os.path.basename(path)}...")
    if not os.path.exists(path):
        print(f"❌ Το αρχείο δεν βρέθηκε: {path}")
        sys.exit(1)
    try:
        df = pd.read_parquet(path, engine='fastparquet')
    except:
        df = pd.read_parquet(path, engine='pyarrow')

    feature_cols = [c for c in df.columns if c.startswith("umap_")]

    # Εντοπισμός Label (αν υπάρχει)
    possible_labels = ['label', 'labels', 'target', 'is_clickbait', 'class']
    label_col = next((c for c in possible_labels if c in df.columns), None)

    X = df[feature_cols].values.astype(np.float32)

    y = None
    if label_col:
        y = df[label_col].values.astype(int)

    return X, y


def recreate_scaler(train_path):
    print("\n⚖️  Ανακατασκευή StandardScaler από τα Training Data...")
    X_train, _ = load_data(train_path)

    scaler = StandardScaler()
    scaler.fit(X_train)  # Μαθαίνουμε το mean/std από το training set
    print("✅ Scaler fitted successfully!")
    return scaler


def evaluate_models():
    mlflow_helper.setup_mlflow(NEW_EXPERIMENT_NAME)

    # 1. Φόρτωση Gold Dataset (Raw)
    print("\n--- Φόρτωση Δεδομένων ---")
    X_gold_raw, y_gold = load_data(CUSTOM_DATA_PATH)

    # 2. Δημιουργία Scaled έκδοσης του Gold Dataset
    # Φορτώνουμε τα train data για να ρυθμίσουμε τον scaler
    scaler = recreate_scaler(TRAIN_DATA_PATH)
    X_gold_scaled = scaler.transform(X_gold_raw)  # Εφαρμόζουμε το scaling στο Gold dataset

    print(f"\n🚀 Έναρξη Αξιολόγησης {len(MODELS_TO_EVALUATE)} Μοντέλων...")

    results = []

    for model_name, config in MODELS_TO_EVALUATE.items():
        model_path = config["path"]
        needs_scaling = config["needs_scaling"]

        print(f"\n🔍 Αξιολόγηση: {model_name} ...")

        if not os.path.exists(model_path):
            print(f"   ❌ Το αρχείο .pkl δεν βρέθηκε: {model_path}")
            continue

        with mlflow.start_run(run_name=f"Eval_{model_name}"):
            try:
                # Φόρτωση μοντέλου
                model = joblib.load(model_path)

                # Επιλογή σωστών δεδομένων (Scaled ή Raw)
                if needs_scaling:
                    print("   ⚖️  Using SCALED data (StandardScaler)")
                    X_input = X_gold_scaled
                else:
                    print("   RAW Using RAW UMAP data (No Scaling)")
                    X_input = X_gold_raw

                # Πρόβλεψη
                preds = model.predict(X_input)

                acc = accuracy_score(y_gold, preds)
                f1 = f1_score(y_gold, preds)

                print(f"   📊 Accuracy: {acc:.4f} | F1: {f1:.4f}")

                mlflow.log_param("model_name", model_name)
                mlflow.log_param("data_scaling", "Scaled" if needs_scaling else "Raw")

                mlflow_helper.evaluate_and_log_metrics(model, X_input, y_gold, prefix="gold")

                results.append({"Model": model_name, "Accuracy": acc, "F1-Score": f1})

            except Exception as e:
                print(f"   ❌ Σφάλμα: {e}")

    if results:
        results_df = pd.DataFrame(results)
        print("\n🏆 Συγκεντρωτικά Αποτελέσματα:")
        print(results_df)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x="Model", y="F1-Score", palette="viridis")
        plt.title("Σύγκριση Μοντέλων (F1 Score)")
        plt.tight_layout()
        plt.savefig("benchmark_results_rescaled.png")


if __name__ == "__main__":
    evaluate_models()