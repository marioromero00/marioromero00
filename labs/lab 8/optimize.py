import mlflow
import mlflow.xgboost
import optuna
import pickle
import os
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from optuna.visualization import plot_optimization_history
from mlflow import log_artifact

#  carpetas necesarias
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# datos para entrenar
df = pd.read_csv("water_potability.csv")

# limpieza muy simplificada
df = df.dropna()

#dropeamos la y
X = df.drop("Potability", axis=1)
y = df["Potability"]

# div de datos
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=1)

# ptimo 
def get_best_model(params):
    return xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")

#  objetivo para Optuna
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }

    with mlflow.start_run(run_name=f"XGBoost con lr {params['learning_rate']:.3f}"):
        model = get_best_model(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        mlflow.log_metric("valid_f1", f1)
        mlflow.log_params(params)
        return f1

# seleccion del mejor y guardado
def optimize_model():
    mlflow.set_experiment("Optuna_XGBoost_Optimization")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    fig = plot_optimization_history(study)
    fig.write_image("plots/optimization_history.png")
    log_artifact("plots/optimization_history.png")

    best_params = study.best_trial.params
    best_model = get_best_model(best_params)
    best_model.fit(X_train_full, y_train_full)

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact("models/best_model.pkl")

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_full)
    shap.summary_plot(shap_values, X_train_full, show=False)
    plt.savefig("plots/feature_importance.png")
    mlflow.log_artifact("plots/feature_importance.png")

    with open("models/requirements.txt", "w") as f:
        import pkg_resources
        dists = [f"{d.project_name}=={d.version}" for d in pkg_resources.working_set]
        f.write("\n".join(dists))
    mlflow.log_artifact("models/requirements.txt")

    print("Optimización completada. Modelo guardado en /models.")

if __name__ == "__main__":
    optimize_model()
