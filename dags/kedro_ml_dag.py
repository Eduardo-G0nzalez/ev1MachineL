"""
Apache Airflow DAG para orquestación de pipelines de Machine Learning.

Este DAG ejecuta:
1. Pipeline de Preparación de Datos
2. Pipeline de EDA
3. Pipeline de Clasificación
4. Pipeline de Regresión
5. Pipeline de Evaluación Final

Autores: Mathias Jara & Eduardo Gonzalez
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os

# Configuración por defecto
default_args = {
    'owner': 'mathias_jara',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Definir el DAG
dag = DAG(
    'kedro_ml_pipeline',
    default_args=default_args,
    description='DAG para ejecutar pipelines de ML con Kedro',
    schedule_interval=None,  # Trigger manual
    catchup=False,
    tags=['kedro', 'machine-learning', 'classification', 'regression'],
)

# ============================================
# TASK 1: Preparar Datos
# ============================================
# Ejecutar en el contenedor de Kedro usando docker exec
# -w /app especifica el directorio de trabajo dentro del contenedor
prepare_data = BashOperator(
    task_id='prepare_data',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=data_preparation_pipeline',
    dag=dag,
)

# ============================================
# TASK 2: Análisis Exploratorio
# ============================================
run_eda = BashOperator(
    task_id='run_eda',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=eda_pipeline',
    dag=dag,
)

# ============================================
# TASK 3: Pipeline de Clasificación
# ============================================
train_classification = BashOperator(
    task_id='train_classification_models',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=classification_pipeline',
    dag=dag,
)

# ============================================
# TASK 4: Pipeline de Regresión
# ============================================
train_regression = BashOperator(
    task_id='train_regression_models',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=regression_pipeline',
    dag=dag,
)

# ============================================
# TASK 5: Evaluación Final
# ============================================
evaluate_models = BashOperator(
    task_id='evaluate_all_models',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=ml_modeling_pipeline',
    dag=dag,
)

# ============================================
# TASK 6: Generar Reporte
# ============================================
generate_report = BashOperator(
    task_id='generate_final_report',
    bash_command='docker exec -w /app ml-letterboxd-pipeline python scripts/generate_report.py',
    dag=dag,
)

# ============================================
# DEFINIENDO DEPENDENCIAS
# ============================================
# Orden de ejecución:
# 1. Prepare data
# 2. EDA (depende de prepare_data)
# 3. Train models (depende de EDA) - paralelo
# 4. Evaluate (depende de ambos trains)
# 5. Report (depende de evaluate)

prepare_data >> run_eda
run_eda >> [train_classification, train_regression]
[train_classification, train_regression] >> evaluate_models
evaluate_models >> generate_report

