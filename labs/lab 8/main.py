from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# cargar modelo entrenado
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Crear app
app = FastAPI(title="Modelo de Potabilidad de Agua")

# Clase para la entrada esperada (puedes ajustar los nombres según tus features reales)
class WaterSample(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float

#  inicio
@app.get("/")
def home():
    return {
        "modelo": "XGBoost optimizado con Optuna",
        "problema": "Clasificación binaria de potabilidad del agua",
        "entrada": "Valores numéricos de características físico-químicas del agua",
        "salida": "0 = no potable, 1 = potable"
    }

# ruta para predecir potabilidad
@app.post("/potabilidad/")
def predict_potability(data: WaterSample):
    features = np.array([[
        data.feature1,
        data.feature2,
        data.feature3,
        data.feature4,
        data.feature5,
        data.feature6,
        data.feature7,
        data.feature8,
        data.feature9
    ]])
    prediction = model.predict(features)[0]
    return {"potabilidad": int(prediction)}
