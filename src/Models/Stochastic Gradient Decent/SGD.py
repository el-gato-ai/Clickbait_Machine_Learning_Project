import pandas as pd
import numpy as np
import optuna
import mlflow
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# --- ΡΥΘΜΙΣΕΙΣ PATHS & MLFLOW HELPER ---
# Βρίσκουμε τον φάκελο που είναι το script και πάμε πίσω για να βρούμε το helper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlflow_helper

# Δυναμικός εντοπισμός των αρχείων δεδομένων
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '../../..'))
DATA_FOLDER_NAME = "merged"
data_path = os.path.join(project_root, 'data', DATA_FOLDER_NAME)

PARQUET_FILE = os.path.join(data_path, "data_merged_embed.parquet")
CSV_FILE = os.path.join(data_path, "data_merged.csv")
TARGET_COL = "label"


def load_and_prep_data():
    print("⏳ Φόρτωση και συγχώνευση δεδομένων...")

    # 1. Φόρτωση Embeddings
    try:
        df_emb = pd.read_parquet(PARQUET_FILE)
    except FileNotFoundError:
        print(f"❌ Το αρχείο {PARQUET_FILE} δεν βρέθηκε.")
        exit()

    # 2. Φόρτωση Labels
    try:
        df_lbl = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"❌ Το αρχείο {CSV_FILE} δεν βρέθηκε.")
        exit()

    # 3. Έλεγχος Συμβατότητας
    if len(df_emb) != len(df_lbl):
        print(f"❌ Σφάλμα: Τα αρχεία δεν ταιριάζουν! Embeddings: {len(df_emb)}, Labels: {len(df_lbl)}")
        exit()

    # 4. Εξαγωγή του X (Embeddings)
    print(f"ℹ️ Το αρχείο Parquet έχει {len(df_emb.columns)} στήλες.")

    if len(df_emb.columns) > 1:
        print("ℹ️ Ανίχνευση πολλαπλών στηλών. Χρήση όλου του DataFrame ως features.")
        X = df_emb.values  
    else:
        col_name = df_emb.columns[0]
        print(f"ℹ️ Ανίχνευση μίας στήλης ('{col_name}'). Μετατροπή λιστών σε numpy array.")
        X = np.stack(df_emb[col_name].values)

    # 5. Εξαγωγή του y (Labels)
    y = df_lbl[TARGET_COL].values.astype(int)

    print(f"✅ Δεδομένα φορτώθηκαν. X Shape: {X.shape}, y Shape: {y.shape}")
    return X, y


def get_data_splits(X, y):
    # 1. Test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # 2. Train (70% αρχικού) και Val (15% αρχικού)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def objective(trial, X_tr, y_tr, X_v, y_v, normalization_status):
    # --- Search Space SGD ---
    loss_type = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber", "perceptron"])
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True) # Learning Rate / Regularization strength

    params = {
        "loss": loss_type,
        "penalty": penalty,
        "alpha": alpha,
        "max_iter": 1000,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5, # Αν δεν βελτιωθεί για 5 φορές, σταμάτα
        "random_state": 42,
        "normalization": normalization_status
    }

    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

    # Φιλτράρισμα παραμέτρων που δεν ανήκουν στον SGDClassifier (όπως το normalization)
    model_params = {k: v for k, v in params.items() if k != 'normalization'}

    model = SGDClassifier(**model_params)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_v)
    f1 = f1_score(y_v, preds)
    acc = accuracy_score(y_v, preds)

    metrics = {"val_f1": f1, "val_accuracy": acc}

    mlflow_helper.log_optuna_trial(trial, params, metrics, model, "sgd_model")

    return f1


def run_experiment_scenario(scenario_name, X_tr, y_tr, X_v, y_v, X_te, y_te, use_norm, scaler_obj=None):
    print(f"\n🚀 Έναρξη σεναρίου: {scenario_name}")

    mlflow_helper.setup_mlflow("Clickbait_SGD_Comparison")

    with mlflow.start_run(run_name=scenario_name) as run:
        mlflow.log_param("normalization_used", use_norm)

        # Sampler
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_v, y_v, str(use_norm)), n_trials=20)

        print(f"🏆 Best params for {scenario_name}: {study.best_params}")

        # --- FINAL TRAINING ---
        print("⚙️ Εκπαίδευση του Champion Model...")
        best_params = study.best_params
        
        # Προσθήκη defaults αν λείπουν (π.χ. random_state)
        if "random_state" not in best_params:
            best_params["random_state"] = 42

        # Το SGDClassifier χρειάζεται προσοχή με το elasticnet ratio
        final_model = SGDClassifier(**best_params)

        # Ένωση Train + Val
        X_full_train = np.concatenate((X_tr, X_v))
        y_full_train = np.concatenate((y_tr, y_v))

        final_model.fit(X_full_train, y_full_train)

        # Αποθήκευση μοντέλου
        mlflow.sklearn.log_model(final_model, artifact_path="champion_model")

        # --- EVALUATION ON TEST SET ---
        print("📈 Υπολογισμός τελικών μετρικών στο Test Set...")

        # Σημαντικό: Αν χρησιμοποιούμε scaler, πρέπει να μετατρέψουμε το Test set!
        if use_norm and scaler_obj is not None:
            X_test_final = scaler_obj.transform(X_te)
        else:
            X_test_final = X_te

        mlflow_helper.evaluate_and_log_metrics(final_model, X_test_final, y_te, prefix="test")

        print(f"✅ Ολοκληρώθηκε. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    if not PARQUET_FILE:
        print("⚠️ ΠΡΟΣΟΧΗ: Δεν έχεις ορίσει τα paths σωστά!")
    else:
        # 1. Φόρτωση
        X, y = load_and_prep_data()
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(X, y)

        # ==========================================
        # SCENARIO 1: RAW DATA (Για σύγκριση - αναμένουμε να πάει χειρότερα)
        # ==========================================
        run_experiment_scenario(
            "SGD_Raw_Data",
            X_train, y_train, X_val, y_val, X_test, y_test,
            use_norm=False,
            scaler_obj=None
        )

        # ==========================================
        # SCENARIO 2: NORMALIZED DATA (Το βασικό)
        # ==========================================
        print("\n⚖️ Εφαρμογή Normalization (StandardScaler)...")
        scaler = StandardScaler()
        
        # Fit μόνο στο Train!
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        # Το Test scaled γίνεται transform αυτόματα μέσα στη συνάρτηση run_experiment_scenario
        
        run_experiment_scenario(
            "SGD_Normalized_Data",
            X_train_scaled, y_train, X_val_scaled, y_val, X_test, y_test,
            use_norm=True,
            scaler_obj=scaler
        )

        print("\n✅ Τέλος!")