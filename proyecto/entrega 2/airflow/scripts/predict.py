import pandas as pd
import joblib
import numpy as np
import os

def generate_predictions(
    model_path,
    data_path,
    id_cols=["customer_id", "product_id"],
    output_path="airflow/output/predictions_positive.csv",
    threshold=0.5
):
    # --- 1. carga modelo ---
    model = joblib.load(model_path)
    print(f" Modelo cargado desde {model_path}")

    # --- 2. leer datos  ---
    df = pd.read_parquet(data_path)

    # --- 3. determinar semana reciente ---
    max_year = df["year"].max()
    max_week = df[df["year"] == max_year]["week"].max()

    print(f"Datos más recientes: año={max_year}, semana={max_week}")

    # filtrar datos de la última semana disponible
    df_latest = df[(df["year"] == max_year) & (df["week"] == max_week)]

    if df_latest.empty:
        print("No hay datos de la última semana. No se generan predicciones.")
        return

    # --- 4. preparar features para predicción ---
    # mantener solo las columnas usadas en el entrenamiento
    feature_cols = ['week', 'year', 'total_items', 'num_orders', 'avg_items_per_order']

    X_pred = df_latest[feature_cols]

    print(f"Shape de X_pred: {X_pred.shape}")
    print(f"Columnas usadas en predicción: {feature_cols}")
    print(f"Tios de datos:\n{X_pred.dtypes}")

    if X_pred.empty:
        print("No hay datos para predecir. Terminando.")
        return

    # --- 5.predicciones ---
    probs = model.predict_proba(X_pred)[:, 1]

    # --- 6. construir dataframe de predicciones ---
    preds_df = df_latest[id_cols].copy()
    preds_df["probability"] = probs

    # filtrar solo predicciones positivas
    preds_positive = preds_df[preds_df["probability"] >= threshold]

    # --- 7. guardar CSV ---
    if not preds_positive.empty:
        preds_positive[id_cols].to_csv(output_path, index=False)
        print(f" Predicciones positivas guardadas en {output_path}")
    else:
        print("sin hay predicciones positivas con el umbral definido.")
