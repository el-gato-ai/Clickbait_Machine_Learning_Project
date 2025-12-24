import pandas as pd
import numpy as np
import optuna
import mlflow
import pickle
from sklearn.ensemble import GradientBoostingClassifier
# Αν θες πιο γρήγορο training, άλλαξε το παραπάνω σε:
# from sklearn.ensemble import HistGradientBoostingClassifier
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


def objective(trial, X_tr, y_tr, X_v, y_v):
    # --- Search Space ---
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42
    }

    model = GradientBoostingClassifier(**params)
    model.fit(X_tr, y_tr)

    # Αξιολόγηση
    preds = model.predict(X_v)
    acc = accuracy_score(y_v, preds)
    f1 = f1_score(y_v, preds)

    metrics = {"val_accuracy": acc, "val_f1": f1}

    # Καταγραφή Trial
    mlflow_helper.log_optuna_trial(trial, params, metrics, model, "gb_model")

    return f1


def run_experiment_scenario(scenario_name, X_tr, y_tr, X_v, y_v, X_te, y_te):
    print(f"\n🚀 Έναρξη σεναρίου: {scenario_name}")

    mlflow_helper.setup_mlflow("Clickbait_GradientBoosting_Comparison")

    with mlflow.start_run(run_name=scenario_name) as run:
        # Δεν έχουμε normalization, άρα το λογκάρουμε ως False
        mlflow.log_param("normalization_used", False)

        # Optuna Setup
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Τρέχουμε το objective (δεν χρειάζεται πλέον το normalization_status argument)
        study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_v, y_v), n_trials=15)

        print(f"🏆 Best params for {scenario_name}: {study.best_params}")

        # --- ΤΕΛΙΚΗ ΕΚΠΑΙΔΕΥΣΗ CHAMPION MODEL ---
        print("⚙️ Εκπαίδευση του Champion Model...")
        best_params = study.best_params
        if "random_state" not in best_params:
            best_params["random_state"] = 42

        final_model = GradientBoostingClassifier(**best_params)

        # Ένωση Train + Val
        X_full_train = np.concatenate((X_tr, X_v))
        y_full_train = np.concatenate((y_tr, y_v))

        final_model.fit(X_full_train, y_full_train)

        # Αποθήκευση του μοντέλου
        mlflow.sklearn.log_model(final_model, artifact_path="champion_model")

        # --- ΤΕΛΙΚΗ ΑΞΙΟΛΟΓΗΣΗ (TEST SET) ---
        print("📈 Υπολογισμός τελικών μετρικών στο Test Set...")
        
        # Εδώ το X_te είναι καθαρό (raw), όπως ακριβώς βγήκε από το split
        mlflow_helper.evaluate_and_log_metrics(final_model, X_te, y_te, prefix="test")
        
        print(f"✅ Ολοκληρώθηκε. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    if not PARQUET_FILE:
        print("⚠️ ΠΡΟΣΟΧΗ: Δεν έχεις ορίσει τα paths σωστά!")
    else:
        # 1. Φόρτωση Δεδομένων
        X, y = load_and_prep_data()
        
        # 2. Διαχωρισμός (Train/Val/Test)
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(X, y)

        # 3. ΕΚΤΕΛΕΣΗ ΜΟΝΟ ΤΟΥ RAW DATA ΣΕΝΑΡΙΟΥ
        run_experiment_scenario(
            scenario_name="GB_Raw_Data",
            X_tr=X_train, y_tr=y_train,
            X_v=X_val, y_v=y_val,
            X_te=X_test, y_te=y_test  # Περνάμε και το Test set μέσα
        )

        print("\n✅ Τέλος εκτέλεσης!")