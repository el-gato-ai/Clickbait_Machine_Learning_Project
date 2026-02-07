import pandas as pd
from pathlib import Path
import numpy as np
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# --- ΡΥΘΜΙΣΕΙΣ PATHS ---
# Προσθήκη του φακέλου src/Models στο path για να βρει το mlflow_helper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Models')))
try:
    import mlflow_helper
except ImportError:
    # Fallback αν τρέχει από άλλο φάκελο
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import mlflow_helper
    except:
        pass # mlflow_helper is optional

# ==========================================
# ⚙️ ΡΥΘΜΙΣΕΙΣ ΧΡΗΣΤΗ
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 1. Path του Gold Dataset (Test set)
CUSTOM_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "umap" / "custom_news_umap_500.parquet"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "clean" / "umap" / "train_umap_500.parquet"

# 3. Τα Paths των .pkl αρχείων
MODELS_TO_EVALUATE = {
    "Gradient_Boosting": {
        "path": PROJECT_ROOT / "mlruns" / "176203313038895818" / "models" / "m-11223ec66e124e829ece6083c7b53cc5" / "artifacts" / "model.pkl"
    },
    "Logistic_Regression_NoScaling": {
        "path": PROJECT_ROOT / "mlruns" / "961020367779049974" / "models" / "m-f5332dd335424d659711f5acb87d7eda" / "artifacts" / "model.pkl",
        "needs_scaling": False
    },
    "SVM_NoScaling": {
        "path": PROJECT_ROOT / "mlruns" / "664524367874882829" / "models" / "m-316b355112e14df284738170b470bf27" / "artifacts" / "ts/model.pkl",
        "needs_scaling": False
    },
    "SGD_Classifier": {
        "path": PROJECT_ROOT / "mlruns" / "236777006947026757" / "models" / "m-022afa5b1f9848768d11c9390253ec71" / "artifacts" / "model.pkl",
        "needs_scaling": False
    },
    "SVM_Scaled": {
        "path": PROJECT_ROOT / "mlruns" / "629225326235206482" / "models" / "m-f917c98e8f4c4e0f895ef460199ce813" / "artifacts" / "model.pkl",
        "needs_scaling": True
    },
    "Logistic_Regression_Scaled": {
        "path": PROJECT_ROOT / "mlruns" / "444470392771103284" / "models" / "m-ad71b18c165c4043b615b5fc234a675d" / "artifacts" / "model.pkl",
        "needs_scaling": True
    },
}

NEW_EXPERIMENT_NAME = "Custom_Dataset_Evaluation"


# ==========================================

def calculate_majority_vote(df):
    """
    Υπολογίζει το τελικό label με βάση την πλειοψηφία των annotators (NG, TK, KB).
    """
    annotators = ['NG', 'TK', 'KB']

    # Έλεγχος αν υπάρχουν οι στήλες
    found_annotators = [col for col in annotators if col in df.columns]

    if len(found_annotators) < 1:
        return None, None

    print(f"   👥 Εντοπίστηκαν Annotators: {found_annotators}")

    # Μετατροπή σε numeric (αν είναι strings '0', '1')
    for col in found_annotators:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Majority Vote Logic
    if len(found_annotators) >= 2:
        # Άθροισμα ψήφων
        votes = df[found_annotators].sum(axis=1)
        # Αν οι ψήφοι είναι περισσότερες από τους μισούς (π.χ. >= 2 για 3 άτομα)
        majority_threshold = len(found_annotators) / 2
        y = (votes > majority_threshold).astype(int).values
        print(f"   🗳️ Majority Vote applied (Threshold > {majority_threshold})")
    else:
        # Αν υπάρχει μόνο ένας annotator, παίρνουμε αυτόν
        y = df[found_annotators[0]].values.astype(int)
        print(f"   👤 Single Annotator used: {found_annotators[0]}")

    return y, "majority_vote"


def load_data(path, is_train=False):
    print(f"   📂 Loading: {os.path.basename(path)}...")
    if not os.path.exists(path):
        print(f"❌ Το αρχείο δεν βρέθηκε: {path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(path, engine='fastparqu # Fallback to pyarrowet')
    except:
        df = pd.read_parquet(path, engine='pyarrow')

    # --- 1. FEATURE DETECTION ---
    # Ψάχνουμε για στήλες που ξεκινάνε με "umap_"
    feature_cols = [c for c in df.columns if str(c).startswith("umap_")]

    # Αν δεν βρεθούν, ψάχνουμε για στήλες που είναι ΑΡΙΘΜΟΙ (π.χ. "0", "1", ... "499")
    if not feature_cols:
        # Φιλτράρουμε στήλες που το όνομά τους είναι αριθμός
        # Προσέχουμε να είναι integer strings ("0", "1") ή integers (0, 1)
        numeric_named_cols = [c for c in df.columns if str(c).isdigit()]

        # Συνήθως τα embeddings είναι οι στήλες 0 έως 499
        if len(numeric_named_cols) >= 50:  # Αν βρούμε πολλές τέτοιες στήλες
            # Τις ταξινομούμε για να είμαστε σίγουροι (0, 1, 2...)
            feature_cols = sorted(numeric_named_cols, key=lambda x: int(x))
            print(f"   ⚠️ Δεν βρέθηκε 'umap_' prefix. Χρήση αριθμητικών στηλών ({len(feature_cols)} dims).")

        # Filter for numeric columns that are not in the exclude list
    # Αν ακόμα δεν βρήκαμε, ψάχνουμε όλες τις float στήλες (έσχατη λύση)
    if not feature_cols:
        exclude = ['NG', 'TK', 'KB', 'label', 'labels', 'target', 'text', 'title']
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_float_dtype(df[c])]

    if not feature_cols:
        raise ValueError(f"❌ Δεν βρέθηκαν features (embeddings) στο αρχείο! Στήλες: {df.columns.tolist()[:10]}...")

    print(f"   ✅ Features detected: {len(feature_cols)} dimensions.")
    X = df[feature_cols].values.astype(np.float32)

    # --- 2. LABEL DETECTION (Annotators) ---
    y = None
    if not is_train:
        # Για το Custom Dataset, κάνουμε Majority Vote
        y, method = calculate_majority_vote(df)

        # Αν αποτύχει το Majority Vote, ψάχνουμε για κλασικό label
        if y is None:
            possible_labels = ['label', 'labels', 'target', 'is_clickbait']
            label_col = next((c for c in possible_labels if c in df.columns), None)
            if label_col:
                y = df[label_col].values.astype(int)
                print(f"   🏷️ Using existing label column: {label_col}")
    else:
        # Για το Train Dataset, ψάχνουμε το κλασικό label
        possible_labels = ['label', 'labels', 'target']
        label_col = next((c for c in possible_labels if c in df.columns), None)
        if label_col:
            y = df[label_col].values.astype(int)

    return X, y


def recreate_scaler(train_path):
    print("\n⚖️  Ανακατασκευή StandardScaler από τα Training Data...")
    # Φορτώνουμε μόνο τα features (is_train=True για να μην ψάχνει annotators)
    X_train, _ = load_data(train_path, is_train=True)

    scaler = StandardScaler()
    scaler.fit(X_train)
    print("✅ Scaler fitted successfully!")
    return scaler


def evaluate_models():
    # Setup MLflow if available
    if 'mlflow_helper' in sys.modules:
        mlflow_helper.setup_mlflow(NEW_EXPERIMENT_NAME)
    else:
        try:
            mlflow.set_experiment(NEW_EXPERIMENT_NAME)
        except:
            pass

    # 1. Φόρτωση Custom Dataset (Test set)
    print("\n--- Φόρτωση Custom Dataset (Greek Annotations) ---")
    X_gold_raw, y_gold = load_data(CUSTOM_DATA_PATH, is_train=False)

    if y_gold is None:
        print("❌ Σφάλμα: Δεν βρέθηκαν labels (NG, TK, KB) στο αρχείο!")
        return

    # 2. Δημιουργία Scaled έκδοσης
    scaler = recreate_scaler(TRAIN_DATA_PATH)

    # Έλεγχος διαστάσεων
    if X_gold_raw.shape[1] != scaler.n_features_in_:
        print(f"❌ Mismatch dimensions! Train: {scaler.n_features_in_}, Custom: {X_gold_raw.shape[1]}")
        print("   Πρέπει να ξανατρέξεις το UMAP στο custom dataset για να βγάλει 500 διαστάσεις.")
        return

    X_gold_scaled = scaler.transform(X_gold_raw)

    print(f"\n🚀 Έναρξη Αξιολόγησης {len(MODELS_TO_EVALUATE)} Μοντέλων...")

    results = []

    for model_name, config in MODELS_TO_EVALUATE.items():
        model_path = config["path"]
        needs_scaling = config["needs_scaling"]

        print(f"\n🔍 Αξιολόγηση: {model_name} ...")

        if not os.path.exists(model_path):
            print(f"   ❌ Το αρχείο .pkl δεν βρέθηκε: {model_path}")
            continue

        try:
            # Φόρτωση μοντέλου
            model = joblib.load(model_path)

            # Επιλογή σωστών δεδομένων
            if needs_scaling:
                print("   ⚖️  Using SCALED data")
                X_input = X_gold_scaled
            else:
                print("   RAW Using RAW data")
                X_input = X_gold_raw

            # Πρόβλεψη
            preds = model.predict(X_input)

            # Metrics
            acc = accuracy_score(y_gold, preds)
            f1 = f1_score(y_gold, preds)
            prec = precision_score(y_gold, preds, zero_division=0)
            rec = recall_score(y_gold, preds, zero_division=0)

            print(f"   📊 Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

            # Log to MLflow
            with mlflow.start_run(run_name=f"CustomEval_{model_name}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("dataset", "Custom_Greek_Annotated")

                mlflow.log_metric("custom_accuracy", acc)
                mlflow.log_metric("custom_f1", f1)
                mlflow.log_metric("custom_precision", prec)
                mlflow.log_metric("custom_recall", rec)

                # Confusion Matrix Plot
                cm = confusion_matrix(y_gold, preds)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title(f"Confusion Matrix: {model_name}")
                plt.ylabel('True Label (Annotators)')
                plt.xlabel('Predicted Label')
                plt.savefig(f"cm_{model_name}.png")
                mlflow.log_artifact(f"cm_{model_name}.png")
                plt.close()

            results.append({
                "Model": model_name,
                "Accuracy": acc,
                "F1-Score": f1,
                "Precision": prec,
                "Recall": rec
            })

        except Exception as e:
            print(f"   ❌ Σφάλμα κατά την εκτέλεση: {e}")

    if results:
        results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
        print("\n🏆 Τελικά Αποτελέσματα στο Custom Dataset:")
        print(results_df.to_string(index=False))

        # Save results to CSV for easy copy-paste
        results_df.to_csv("custom_eval_results.csv", index=False)
        print("\n✅ Τα αποτελέσματα αποθηκεύτηκαν στο 'custom_eval_results.csv'")


if __name__ == "__main__":
    evaluate_models()