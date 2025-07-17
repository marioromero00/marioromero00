from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import pandas as pd
import joblib
import os
import sys

sys.path.append("/opt/airflow/scripts")

# scripts
from extract_data import extract_data
from preprocess_data import preprocess
from detect_drift import detect_drift
from train_model import train_xgb_model
from predict import generate_predictions

default_args = {
    'owner': 'mario',
    'retries': 1,
}

# DAG 
with DAG(
    dag_id='predictive_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    default_args=default_args,
    catchup=False,
) as dag:

  

    def extract_task(**kwargs):
        data_dir = Variable.get("DATA_DIR")
        df = extract_data(data_dir, filename="transaccion.parquet")
        df.to_parquet("/opt/airflow/output/raw_data.parquet")
        print("Datos extraídos y guardados.")



    def preprocess_task(**kwargs):
        df_transacciones = pd.read_parquet("/opt/airflow/output/raw_data.parquet")
        df_productos = pd.read_parquet("/opt/airflow/data/productos.parquet")
        df_clientes = pd.read_parquet("/opt/airflow/data/clientes.parquet")

        df_clean = preprocess(df_transacciones, df_productos, df_clientes)
        df_clean["target"] = (df_clean["total_items"] > 0).astype(int)

        print("Conteo de target:\n", df_clean["target"].value_counts())

        df_clean.to_parquet("/opt/airflow/output/preprocessed_data.parquet")
        print(" Datos preprocesados y guardados.")

        ref_path = "/opt/airflow/output/reference_data.parquet"
        if not os.path.exists(ref_path):
            df_clean.to_parquet(ref_path)
            print("Archivo de referencia guardado.")

    
   #deteccion de dir en las sgtes features
    def detect_drift_task(**kwargs):
        ref_df = pd.read_parquet("/opt/airflow/output/reference_data.parquet")
        new_df = pd.read_parquet("/opt/airflow/output/preprocessed_data.parquet")

        features = ["total_items", "num_orders", "avg_items_per_order"]

        drift_detected, p_values = detect_drift(ref_df, new_df, features)

        print(f"drift detected: {drift_detected}")
        print(f"P-values: {p_values}")

        kwargs['ti'].xcom_push(key='drift_detected', value=drift_detected)

   

    def branch_drift(**kwargs):
        model_path = "/opt/airflow/output/final_model.pkl"
        model_exists = os.path.exists(model_path)

        drift_detected = kwargs['ti'].xcom_pull(key='drift_detected')

        if not model_exists:
            print(" No existe modelo entrenado. Procediendo a entrenar.")
            return 'train_model'

        if drift_detected:
            print(" Drift detectado. Procediendo a entrenar.")
            return 'train_model'
        else:
            print("No hay drift y ya existe modelo. Se salta el entrenamiento.")
            return 'skip_training'

 

    def train_model_task(**kwargs):
        df = pd.read_parquet("/opt/airflow/output/preprocessed_data.parquet")

        model = train_xgb_model(
            data=df,
            mlflow_tracking_uri=Variable.get("MLFLOW_TRACKING_URI"),
            experiment_name="entrega2_pipeline",
            target_col="target",
            n_trials=10
        )
        print(" Modelo entrenado y guardado.")

 

    def skip_training_task(**kwargs):
        print(" Drift NO detectado. Se salta reentrenamiento.")



    def integrate_batch_parquet_task(**kwargs):
        
        batch_filename = Variable.get("NEW_BATCH_FILENAME")

        # 1. leer nuevo archivo 
        df_new_pairs = pd.read_parquet(f"/opt/airflow/data/{batch_filename}")

        # 2. cargar 
        df_hist = pd.read_parquet("/opt/airflow/output/preprocessed_data.parquet")

        # 3. filtrar last semana
        max_year = df_hist["year"].max()
        max_week = df_hist[df_hist["year"] == max_year]["week"].max()

        df_latest = df_hist[
            (df_hist["year"] == max_year) &
            (df_hist["week"] == max_week)
        ]

        # 4. hacer join para enriquecer con nueva dara
        df_new_week = df_new_pairs.merge(
            df_latest,
            on=["customer_id", "product_id"],
            how="left"
        )

        df_new_week.fillna(0, inplace=True)

        # 5. asignar nueva semana
        df_new_week["week"] = max_week + 1
        df_new_week["year"] = max_year

        # 6. quitar target 
        if "target" in df_new_week.columns:
            df_new_week.drop(columns=["target"], inplace=True)

        # 7. concatenar al hist
        df_hist_updated = pd.concat(
            [df_hist, df_new_week],
            ignore_index=True
        )

        # eliminar duplicados (opr si subia la misma data dos veces)
        df_hist_updated = df_hist_updated.drop_duplicates(
            subset=["year", "week", "customer_id", "product_id"],
            keep='last'
        )

        # guardar historico actualizado
        df_hist_updated.to_parquet(
            "/opt/airflow/output/preprocessed_data.parquet",
            index=False
        )

        # 8. crear parquet para la prox semana
        df_pred_next_week = df_new_week.copy()
        df_pred_next_week["week"] += 1

        if "target" in df_pred_next_week.columns:
            df_pred_next_week.drop(columns=["target"], inplace=True)

        df_pred_next_week.to_parquet(
            "/opt/airflow/output/pred_data_next_week.parquet",
            index=False
        )

        print(f" Integrado {batch_filename} y parquet pred_data_next_week creado.")



    def predict_task(**kwargs):
        generate_predictions(
            model_path="/opt/airflow/output/final_model.pkl",
            data_path="/opt/airflow/output/pred_data_next_week.parquet",
            output_path="/opt/airflow/output/predictions_next_week.csv",
            threshold=0.5
        )
        print("Predicciones generadas para la próxima semana.")

    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_task,
    )

    preprocess_op = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_task,
    )

    drift_op = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift_task,
    )

    branch_op = BranchPythonOperator(
        task_id='branch_drift_decision',
        python_callable=branch_drift,
    )

    train_op = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )

    skip_op = PythonOperator(
        task_id='skip_training',
        python_callable=skip_training_task,
        trigger_rule=TriggerRule.NONE_FAILED
    )

    integrate_batch_parquet_op = PythonOperator(
        task_id='integrate_batch_parquet',
        python_callable=integrate_batch_parquet_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    predict_op = PythonOperator(
        task_id='generate_predictions',
        python_callable=predict_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )



    extract >> preprocess_op >> drift_op >> branch_op
    branch_op >> [train_op, skip_op]
    [train_op, skip_op] >> integrate_batch_parquet_op >> predict_op
