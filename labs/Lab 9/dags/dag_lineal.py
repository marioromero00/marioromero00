from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import os

from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

# URL del CSV
DATA_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"

default_args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='hiring_lineal',
    default_args=default_args,
    description='Pipeline de contratación con modelo lineal',
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,
    catchup=False,
    tags=['contratacion'],
) as dag:

    inicio = EmptyOperator(task_id='inicio')

    crear_carpetas = PythonOperator(
        task_id='crear_carpetas',
        python_callable=create_folders,
        op_kwargs={'ds': '{{ ds }}'},
    )

    descargar_dataset = PythonOperator(
        task_id='descargar_dataset',
        python_callable=lambda ds, **kwargs: os.system(
            f"curl -o dags/{ds}/raw/data_1.csv {DATA_URL}"
        ),
        op_kwargs={'ds': '{{ ds }}'},
    )

    dividir_datos = PythonOperator(
        task_id='dividir_datos',
        python_callable=split_data,
        op_kwargs={'ds': '{{ ds }}'},
    )

    entrenar_modelo = PythonOperator(
        task_id='entrenar_modelo',
        python_callable=preprocess_and_train,
        op_kwargs={'ds': '{{ ds }}'},
    )

    interfaz_gradio = PythonOperator(
        task_id='interfaz_gradio',
        python_callable=gradio_interface,
        op_kwargs={'ds': '{{ ds }}'},
    )

    fin = EmptyOperator(task_id='fin')

    # Flujo de tareas
    inicio >> crear_carpetas >> descargar_dataset >> dividir_datos >> entrenar_modelo >> interfaz_gradio >> fin
