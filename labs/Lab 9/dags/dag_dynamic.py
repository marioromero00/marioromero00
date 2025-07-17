from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta, timezone
import os

# Importar funciones
from hiring_dynamic_functions import create_folders, load_and_merge, split_data, train_model, evaluate_models

# Importar modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Configuración base del DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dynamic_training_pipeline',
    default_args=default_args,
    description='DAG dinámico para entrenamiento de modelos',
    schedule_interval='0 15 5 * *',  # Día 5 de cada mes a las 15:00 UTC
    start_date=datetime(2024, 10, 1),
    catchup=True,  # Permite backfill
    tags=['training', 'branching'],
) as dag:

    # 1. Inicio del pipeline
    start = EmptyOperator(task_id='start')

    # 2. Crear carpetas para esta fecha de ejecución
    create_folder_task = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={'ds': '{{ ds }}'},
    )

    # 3. Branching: lógica para decidir qué archivos descargar
    def choose_branch(**kwargs):
        execution_date = kwargs['execution_date']
        cutoff_date = datetime(2024, 11, 1, tzinfo=timezone.utc)
        print(f"[branching_logic] Execution date: {execution_date} | Cutoff: {cutoff_date}")
        if execution_date < cutoff_date:
            return 'download_data_1'
        else:
            return 'download_data_1_and_2'

    branching = BranchPythonOperator(
        task_id='branching_logic',
        python_callable=choose_branch,
        provide_context=True
    )

    # 4. Descarga de datos
    download_data_1 = PythonOperator(
        task_id='download_data_1',
        python_callable=load_and_merge,
        op_kwargs={'execution_date': '{{ ds }}', 'datasets': ['data_1']},
    )

    download_data_1_and_2 = PythonOperator(
        task_id='download_data_1_and_2',
        python_callable=load_and_merge,
        op_kwargs={'execution_date': '{{ ds }}', 'datasets': ['data_1', 'data_2']},
    )

    # 5. Unir los datos descargados
    merge_task = PythonOperator(
        task_id='merge_data',
        python_callable=load_and_merge,
        op_kwargs={'execution_date': '{{ ds }}'},
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    # 6. Separar en train/test
    split_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'execution_date': '{{ ds }}'},
    )

    # 7. Entrenamiento de modelos
    model_rf = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_model,
        op_kwargs={
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'target_col': 'HiringDecision',
            'ds': '{{ ds }}',
        },
    )

    model_xgb = PythonOperator(
        task_id='train_xgboost',
        python_callable=train_model,
        op_kwargs={
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'target_col': 'HiringDecision',
            'ds': '{{ ds }}',
        },
    )

    model_lr = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=train_model,
        op_kwargs={
            'model': LogisticRegression(max_iter=1000),
            'target_col': 'HiringDecision',
            'ds': '{{ ds }}',
        },
    )

    # 8. Evaluación del mejor modelo
    evaluate = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        op_kwargs={'ds': '{{ ds }}'},
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # 9. Fin del pipeline
    end = EmptyOperator(task_id='end')

    # Encadenamiento de tareas
    start >> create_folder_task >> branching
    branching >> download_data_1 >> merge_task
    branching >> download_data_1_and_2 >> merge_task
    merge_task >> split_task >> [model_rf, model_xgb, model_lr] >> evaluate >> end
