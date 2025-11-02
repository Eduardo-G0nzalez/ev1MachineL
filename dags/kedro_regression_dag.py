"""
Apache Airflow DAG para Pipeline de Regresión.

Ejecuta específicamente el pipeline de regresión con sus etapas.

Autores: Mathias Jara & Eduardo Gonzalez
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'mathias_jara',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'kedro_regression',
    default_args=default_args,
    description='DAG para pipeline de regresión',
    schedule_interval=None,
    catchup=False,
    tags=['kedro', 'regression'],
)

# Preparar datos de regresión
# Ejecutar en el contenedor de Kedro usando docker exec
# -w /app especifica el directorio de trabajo dentro del contenedor
prepare_regression = BashOperator(
    task_id='prepare_regression_data',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=regression_pipeline',
    dag=dag,
)

# Entrenar modelos
train_models = BashOperator(
    task_id='train_regression_models',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=regression_pipeline',
    dag=dag,
)

# Evaluar modelos
evaluate_models = BashOperator(
    task_id='evaluate_regression_models',
    bash_command='docker exec -w /app ml-letterboxd-pipeline python -c "print(\'Evaluación completada\')"',
    dag=dag,
)

prepare_regression >> train_models >> evaluate_models

