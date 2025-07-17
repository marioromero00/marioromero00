import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import os
import joblib

def train_xgb_model(
    data,
    mlflow_tracking_uri,
    experiment_name,
    target_col="target",
    id_cols=["customer_id", "product_id"],
    test_size=0.3, 
    random_state=42,
    n_trials=20,
    save_local_model=True,
    local_model_path="/opt/airflow/output/final_model.pkl"
):
    # --- 1. separar variables ---
    cols_to_drop = id_cols + [target_col]
    feature_cols = [col for col in data.columns if col not in cols_to_drop]

    X = data[feature_cols]
    y = data[target_col]

    # --- 2. spliteamos ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # --- 3. otimizamos  ---
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }

        clf = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1,
            **params
        )

        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(" Best hyperparameters found:", best_params)

    # --- 4. entrenamos modelo ---
    final_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1,
        **best_params
    )
    final_model.fit(X, y)

    # --- 5. MLflow  ---
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        # kog hyperparameters
        mlflow.log_params(best_params)

        # metricas a reportar
        preds_train = final_model.predict_proba(X_train)[:, 1]
        preds_val = final_model.predict_proba(X_val)[:, 1]

        auc_train = roc_auc_score(y_train, preds_train)
        auc_val = roc_auc_score(y_val, preds_val)

        acc_train = accuracy_score(y_train, final_model.predict(X_train))
        acc_val = accuracy_score(y_val, final_model.predict(X_val))

        mlflow.log_metric("auc_train", auc_train)
        mlflow.log_metric("auc_val", auc_val)
        mlflow.log_metric("accuracy_train", acc_train)
        mlflow.log_metric("accuracy_val", acc_val)

        print(f" AUC train: {auc_train:.4f}")
        print(f" AUC validation: {auc_val:.4f}")

        # --- SHAP ---
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_val)

        #  guardar en ruta compartida con ML
        shap_path = "/mlflow/mlruns/shap_summary.png"
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X_val, show=False)
        plt.savefig(shap_path)
        mlflow.log_artifact(shap_path)
        plt.close()

        # Log model to MLflow
        mlflow.xgboost.log_model(final_model, "xgb_model")

    # --- 6. guardar model---
    if save_local_model:
        joblib.dump(final_model, local_model_path)
        print(f" Modelo guardado localmente en {local_model_path}")

    return final_model
