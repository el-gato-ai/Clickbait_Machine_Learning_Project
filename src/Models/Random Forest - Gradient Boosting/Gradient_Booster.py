import pandas as pd
import numpy as np
import optuna
import mlflow
import pickle  # <--- Απαραίτητο για την αποθήκευση του Scaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow_helper

# --- ΡΥΘΜΙΣΕΙΣ ---
PARQUET_FILE = ""  # <-- Βάλε το σωστό path
EMBEDDING_COL = ""  # <-- Όνομα στήλης embeddings
TARGET_COL = ""  # <-- Όνομα στήλης στόχου (0/1)


def load_and_prep_data():
    print("⏳ Φόρτωση Parquet αρχείου...")
    try:
        df = pd.read_parquet(PARQUET_FILE)
    except FileNotFoundError:
        print(f"❌ Το αρχείο {PARQUET_FILE} δεν βρέθηκε. Ελεγξε το path.")
        exit()
    except Exception as e:
        print(f"❌ Κάτι πήγε στραβά με τη φόρτωση: {e}")
        exit()

    # Μετατροπή της στήλης embeddings (που είναι λίστα) σε 2D numpy array
    X = np.stack(df[EMBEDDING_COL].values)
    y = df[TARGET_COL].values

    print(f"✅ Δεδομένα φορτώθηκαν. Shape: {X.shape}")
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
    # --- ΥΠΕΡ-ΠΑΡΑΜΕΤΡΟΙ ΓΙΑ GRADIENT BOOSTING ---
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        # Το random_state βοηθάει να είναι σταθερά τα αποτελέσματα
        "random_state": 42,
        "normalization": normalization_status
    }

    # Φιλτράρουμε το 'normalization' πριν το περάσουμε στο μοντέλο
    model_params = {k: v for k, v in params.items() if k != 'normalization'}

    model = GradientBoostingClassifier(**model_params)
    model.fit(X_tr, y_tr)

    # Αξιολόγηση
    preds = model.predict(X_v)
    acc = accuracy_score(y_v, preds)
    f1 = f1_score(y_v, preds)

    metrics = {"val_accuracy": acc, "val_f1": f1}

    # Καταγραφή Trial
    mlflow_helper.log_optuna_trial(trial, params, metrics, model, "gb_model")

    return f1


def run_experiment_scenario(scenario_name, X_tr, y_tr, X_v, y_v, use_norm, scaler_obj=None):
    print(f"\n🚀 Έναρξη σεναρίου: {scenario_name}")

    mlflow_helper.setup_mlflow("Clickbait_GradientBoosting_Comparison")

    with mlflow.start_run(run_name=scenario_name) as run:
        mlflow.log_param("normalization_used", use_norm)

        # Sampler για σταθερά αποτελέσματα (reproducibility)
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_v, y_v, str(use_norm)), n_trials=15)

        print(f"🏆 Best params for {scenario_name}: {study.best_params}")

        # --- ΤΕΛΙΚΗ ΕΚΠΑΙΔΕΥΣΗ CHAMPION MODEL ---
        print("⚙️ Εκπαίδευση του Champion Model...")
        best_params = study.best_params

        # Προσθέτουμε το random_state και εδώ για σιγουριά, αν δεν το έβγαλε το optuna
        if "random_state" not in best_params:
            best_params["random_state"] = 42

        final_model = GradientBoostingClassifier(**best_params)

        # Ένωση Train + Val
        X_full_train = np.concatenate((X_tr, X_v))
        y_full_train = np.concatenate((y_tr, y_v))

        # ... (ο κώδικας που είχες για το fit του final_model) ...
        final_model.fit(X_full_train, y_full_train)

        # Ε. Αποθήκευση του μοντέλου
        mlflow.sklearn.log_model(final_model, artifact_path="champion_model")

        # --- ΝΕΟ ΚΟΜΜΑΤΙ: ΠΛΗΡΗΣ ΑΞΙΟΛΟΓΗΣΗ ---
        # Καλούμε τη νέα συνάρτηση από το helper
        # Χρησιμοποιούμε το X_test που είχαμε κρατήσει στην άκρη και δεν το ακούμπησε κανείς!
        print("📈 Υπολογισμός τελικών μετρικών στο Test Set...")

        # ΠΡΟΣΟΧΗ: Αν έχεις scaler, πρέπει να μετατρέψεις το Test set!
        if use_norm and scaler_obj is not None:
            # Χρησιμοποιούμε τον scaler που μόλις εκπαιδεύσαμε/χρησιμοποιήσαμε
            X_test_final = scaler_obj.transform(X_test)
        else:
            X_test_final = X_test

        # Εδώ γίνεται η καταγραφή όλων των γραφημάτων και metrics
        mlflow_helper.evaluate_and_log_metrics(final_model, X_test_final, y_test, prefix="test")

        print(f"✅ Ολοκληρώθηκε. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    if not PARQUET_FILE:
        print("⚠️ ΠΡΟΣΟΧΗ: Δεν έχεις ορίσει το PARQUET_FILE, EMBEDDING_COL ή TARGET_COL!")
    else:
        # 1. Φόρτωση και Split
        X, y = load_and_prep_data()
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(X, y)

        # ==========================================
        # ΠΕΡΙΠΤΩΣΗ 1: ΧΩΡΙΣ NORMALIZATION
        # ==========================================
        run_experiment_scenario(
            scenario_name="GB_Raw_Data",
            X_tr=X_train, y_tr=y_train,
            X_v=X_val, y_v=y_val,
            use_norm=False,
            scaler_obj=None  # Δεν υπάρχει scaler εδώ
        )

        # ==========================================
        # ΠΕΡΙΠΤΩΣΗ 2: ΜΕ NORMALIZATION
        # ==========================================
        print("\n⚖️ Εφαρμογή Normalization (StandardScaler)...")
        scaler = StandardScaler()

        # Fit μόνο στο Train!
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        run_experiment_scenario(
            scenario_name="GB_Normalized_Data",
            X_tr=X_train_scaled, y_tr=y_train,
            X_v=X_val_scaled, y_v=y_val,
            use_norm=True,
            scaler_obj=scaler  # Περνάμε τον scaler για αποθήκευση
        )

        print("\n✅ Ολοκληρώθηκαν και τα δύο σενάρια. Έλεγξε το MLflow UI!")