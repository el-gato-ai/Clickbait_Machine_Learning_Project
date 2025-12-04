import pandas as pd
import numpy as np
import optuna
import mlflow
import pickle  # <--- Χρειάζεται για να σώσουμε τον Scaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow_helper

# --- ΡΥΘΜΙΣΕΙΣ ---
# Μην ξεχάσεις να τα συμπληρώσεις!
PARQUET_FILE = "dataset.parquet"
EMBEDDING_COL = "embeddings"
TARGET_COL = "clickbait"


def load_and_prep_data():
    print("⏳ Φόρτωση δεδομένων...")
    try:
        df = pd.read_parquet(PARQUET_FILE)
    except FileNotFoundError:
        print(f"❌ Το αρχείο {PARQUET_FILE} δεν βρέθηκε.")
        exit()

    X = np.stack(df[EMBEDDING_COL].values)
    y = df[TARGET_COL].values

    print(f"✅ Loaded: {X.shape}")
    return X, y


def get_data_splits(X, y):
    # Split 1: Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Split 2: Val (15% of original -> ~17.65% of temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def objective(trial, X_tr, y_tr, X_v, y_v, normalization_status):
    # --- Search Space ---
    loss_type = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber", "perceptron"])
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)

    params = {
        "loss": loss_type,
        "penalty": penalty,
        "alpha": alpha,
        "max_iter": 1000,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42,
        "normalization": normalization_status
    }

    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

    # Φιλτράρισμα παραμέτρων που δεν ανήκουν στον SGDClassifier
    model_params = {k: v for k, v in params.items() if k != 'normalization'}

    model = SGDClassifier(**model_params)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_v)
    f1 = f1_score(y_v, preds)
    acc = accuracy_score(y_v, preds)

    metrics = {"val_f1": f1, "val_accuracy": acc}

    mlflow_helper.log_optuna_trial(trial, params, metrics, model, "sgd_model")

    return f1


# --- ΔΙΟΡΘΩΣΗ: Προσθέσαμε το όρισμα scaler_obj=None ---
def run_experiment_scenario(scenario_name, X_tr, y_tr, X_v, y_v, use_norm, scaler_obj=None):
    print(f"\n🚀 Έναρξη σεναρίου: {scenario_name}")

    mlflow_helper.setup_mlflow("Clickbait_SGD_Comparison")

    with mlflow.start_run(run_name=scenario_name) as run:
        mlflow.log_param("normalization_used", use_norm)

        # Προσθήκη Seed για να είναι πάντα ίδια τα trials
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_v, y_v, str(use_norm)), n_trials=20)

        print(f"🏆 Best params for {scenario_name}: {study.best_params}")

        # --- FINAL TRAINING ---
        print("⚙️ Εκπαίδευση του Champion Model...")
        best_params = study.best_params

        # Προσοχή: Το SGDClassifier έχει default l1_ratio=0.15.
        # Αν το Optuna διάλεξε 'elasticnet', το l1_ratio θα είναι στο best_params.
        # Αν διάλεξε 'l2', δεν θα είναι, άρα θα πάρει το default (που αγνοείται στο l2). Οπότε είναι ΟΚ.
        final_model = SGDClassifier(**best_params)

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
    # Έλεγχος αν ορίστηκαν τα paths
    if not PARQUET_FILE:
        print("⚠️ ΠΡΟΣΟΧΗ: Δεν έχεις ορίσει το PARQUET_FILE στην αρχή του script!")
    else:
        X, y = load_and_prep_data()
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(X, y)

        # Σενάριο 1: Raw
        run_experiment_scenario(
            "SGD_Raw_Data",
            X_train, y_train, X_val, y_val,
            use_norm=False,
            scaler_obj=None  # Δεν υπάρχει scaler εδώ
        )

        # Σενάριο 2: Normalized
        print("\n⚖️ Εφαρμογή Normalization (StandardScaler)...")
        scaler = StandardScaler()
        # Fit μόνο στο Train!
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        # (Το Test δεν το χρησιμοποιούμε εδώ, αλλά θα έπρεπε να γίνει transform αν το θέλαμε)

        run_experiment_scenario(
            "SGD_Normalized_Data",
            X_train_scaled, y_train, X_val_scaled, y_val,
            use_norm=True,
            scaler_obj=scaler  # <--- Περνάμε τον scaler για να αποθηκευτεί
        )

        print("\n✅ Τέλος!")