import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# Establecer ruta de logs para MLflow
os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")
mlflow.autolog()

# Cargar datos
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Modelo
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)

# Ejecutar run
with mlflow.start_run():
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
