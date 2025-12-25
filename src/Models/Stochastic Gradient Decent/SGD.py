import pandas as pd
import numpy as np
import optuna
import mlflow
import sys
import os
import time
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- ΡΥΘΜΙΣΕΙΣ PATHS ---
# Βρίσκουμε τον φάκελο που είναι το script και πάμε πίσω για να βρούμε το helper
# (Υποθέτουμε ότι το script τρέχει από τον φάκελο 'Models/Stochastic Gradient Decent')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlflow_helper

# Δυναμικός εντοπισμός των αρχείων δεδομένων
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '../../..'))
DATA_FOLDER = os.path.join(project_root, 'data', 'clean', 'umap')


def load_split_data(data_path):
    """
    Φορτώνει τα έτοιμα Train/Valid/Test αρχεία Parquet.
    - Διαχειρίζεται αυτόματα fastparquet/pyarrow.
    - Αναγνωρίζει τα features και το label.
    - Δεν εφαρμόζει scaling (βάσει οδηγίας).
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

        # 1. Φόρτωση DataFrame (Δοκιμή fastparquet -> pyarrow)
        try:
            df = pd.read_parquet(file_path, engine='fastparquet')
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine='pyarrow')
            except Exception as e:
                print(f"⛔ Σφάλμα κατά την ανάγνωση του {filename}: {e}")
                sys.exit(1)

        # 2. Εντοπισμός Features (X)
        feature_cols = [c for c in df.columns if c.startswith("umap_")]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c not in possible_label_cols]

        # 3. Εντοπισμός Labels (y)
        label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break

        # Fallback: Αν δεν βρεθεί label, ψάχνουμε για εξωτερικό αρχείο ή την εναπομείνασα στήλη
        if label_col is None:
            remaining = [c for c in df.columns if c not in feature_cols]
            if len(remaining) == 1:
                label_col = remaining[0]
                print(f"   ⚠️ {name}: Αυτόματος εντοπισμός label στήλης: '{label_col}'")
            else:
                # Έλεγχος για εξωτερικό αρχείο labels
                prefix = filename.split('_')[0]
                ext_label_path = os.path.join(data_path, f"{prefix}_labels.csv")
                if os.path.exists(ext_label_path):
                    print(f"   ℹ️ {name}: Ανάγνωση labels από εξωτερικό αρχείο ({prefix}_labels.csv)")
                    df_labels = pd.read_csv(ext_label_path)
                    y = df_labels.iloc[:, 0].values.astype(int)
                    # Εδώ πρέπει να ορίσουμε το X και να συνεχίσουμε
                    X = df[feature_cols].values.astype(np.float32)
                    loaded_data[name] = (X, y)
                    print(f"   ✅ {name} loaded: X={X.shape}, y={y.shape}")
                    continue
                else:
                    print(f"⛔ Σφάλμα στο {name}: Δεν βρέθηκε στήλη label.")
                    sys.exit(1)

        if label_col:
            if label_col in feature_cols:
                feature_cols.remove(label_col)
            y = df[label_col].values.astype(int)

        # 4. Μετατροπή σε Numpy Arrays
        X = df[feature_cols].values.astype(np.float32)

        if len(X) != len(y):
            print(f"❌ Ασυμφωνία διαστάσεων στο {name}: X={len(X)}, y={len(y)}")
            sys.exit(1)

        loaded_data[name] = (X, y)
        print(f"   ✅ {name} loaded: X={X.shape}, y={y.shape}")

    return loaded_data["Train"], loaded_data["Valid"], loaded_data["Test"]


def objective(trial, X_tr, y_tr, X_v, y_v):
    """
    Objective function για το Optuna (SGD).
    """
    # --- Search Space SGD ---
    loss_type = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber", "perceptron"])
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)

    params = {
        "loss": loss_type,
        "penalty": penalty,
        "alpha": alpha,
        "max_iter": 1000,
        "early_stopping": True,
        "n_iter_no_change": 5,
        "random_state": 42
    }

    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

    model = SGDClassifier(**params)

    # --- Μέτρηση Χρόνου Εκπαίδευσης (Trial) ---
    start_time = time.time()
    model.fit(X_tr, y_tr)
    training_time = time.time() - start_time
    # -------------------------------------------

    # Validation
    preds = model.predict(X_v)
    f1 = f1_score(y_v, preds)
    acc = accuracy_score(y_v, preds)

    metrics = {"val_f1": f1, "val_accuracy": acc, "training_time_sec": training_time}

    # Log στο MLflow με συγκεκριμένο prefix ονόματος
    mlflow_helper.log_optuna_trial(
        trial,
        params,
        metrics,
        model,
        run_name_prefix="SGD_Trial"
    )

    return f1


def run_experiment():
    EXPERIMENT_NAME = "Clickbait_SGD_UMAP_Final_NoScaling"

    # 1. Setup MLflow
    mlflow_helper.setup_mlflow(EXPERIMENT_NAME)
    print(f"\n🚀 Έναρξη Πειράματος: {EXPERIMENT_NAME}")

    # 2. Φόρτωση Δεδομένων
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_split_data(DATA_FOLDER)

    print("ℹ️ Σημείωση: Δεν εφαρμόζεται Feature Scaling (StandardScaler) στα UMAP embeddings.")

    # --- ΦΑΣΗ 1: Hyperparameter Tuning ---
    print("\n🔍 ΦΑΣΗ 1: Αναζήτηση Βέλτιστων Παραμέτρων (Optuna)...")

    # Parent Run για το Tuning
    with mlflow.start_run(run_name="🔍_SGD_Hyperparameter_Tuning") as tuning_run:
        mlflow.log_param("dataset", "UMAP_500")
        mlflow.log_param("scaling", "None")

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Εκτέλεση 20 trials
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)

        print(f"🏆 Best Params found: {study.best_params}")
        print(f"🏆 Best Val F1: {study.best_value:.4f}")

    # --- ΦΑΣΗ 2: Champion Model Training (ΞΕΧΩΡΙΣΤΟ RUN) ---
    print("\n👑 ΦΑΣΗ 2: Εκπαίδευση & Αποθήκευση Champion Model...")

    # Ξεχωριστό Run για το τελικό μοντέλο
    with mlflow.start_run(run_name="👑_SGD_Champion_Model") as final_run:
        # Καταγράφουμε τις παραμέτρους του νικητή
        best_params = study.best_params
        best_params.update({
            "max_iter": 1000,
            "early_stopping": True,
            "n_iter_no_change": 5,
            "random_state": 42
        })
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "SGD_Champion")
        mlflow.log_param("dataset", "UMAP_500")

        final_model = SGDClassifier(**best_params)

        # Ένωση των Train + Valid (χωρίς scaling)
        X_full_train = np.concatenate((X_train, X_val))
        y_full_train = np.concatenate((y_train, y_val))

        # Μέτρηση χρόνου τελικής εκπαίδευσης
        start_t = time.time()
        final_model.fit(X_full_train, y_full_train)
        final_train_time = time.time() - start_t
        print(f"⏱️ Training Time: {final_train_time:.2f} sec")

        # Log του μοντέλου
        mlflow.sklearn.log_model(final_model, artifact_path="champion_model")

        # Evaluation στο Test set
        print("📈 Αξιολόγηση στο Test Set...")
        mlflow_helper.evaluate_and_log_metrics(
            final_model,
            X_test,
            y_test,
            prefix="test",
            training_time=final_train_time
        )

        print(f"\n✅ ΤΕΛΟΣ! Το Champion Model αποθηκεύτηκε στο Run ID: {final_run.info.run_id}")
        print(f"   👉 Αναζητήστε στο MLflow UI το Run με όνομα: '👑_SGD_Champion_Model'")


if __name__ == "__main__":
    run_experiment()