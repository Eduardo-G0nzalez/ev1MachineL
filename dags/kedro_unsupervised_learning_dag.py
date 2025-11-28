"""
Apache Airflow DAG para orquestación del pipeline de Aprendizaje No Supervisado.

Este DAG ejecuta:
1. Pipeline de Aprendizaje No Supervisado completo
   - Clustering (K-Means, DBSCAN, Hierarchical)
   - Reducción de Dimensionalidad (PCA, t-SNE, UMAP)
   - Evaluación y comparación de modelos

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
    'kedro_unsupervised_learning',
    default_args=default_args,
    description='DAG para ejecutar pipeline de Aprendizaje No Supervisado con Kedro',
    schedule_interval=None,  # Trigger manual
    catchup=False,
    tags=['kedro', 'machine-learning', 'unsupervised', 'clustering', 'dimensionality-reduction'],
)

# ============================================
# TASK 1: Pipeline de Aprendizaje No Supervisado Completo
# ============================================
# Este task ejecuta todo el pipeline de unsupervised learning:
# - Preparación de datos para clustering
# - Búsqueda de k óptimo
# - Entrenamiento de K-Means, DBSCAN y Clustering Jerárquico
# - Análisis completo de PCA (varianza, loadings, biplot)
# - Reducción dimensional con t-SNE y UMAP
# - Evaluación y comparación de todos los modelos
unsupervised_learning = BashOperator(
    task_id='run_unsupervised_learning_pipeline',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=unsupervised_learning_pipeline',
    dag=dag,
    execution_timeout=timedelta(hours=2),  # Timeout de 2 horas para el pipeline completo
)

# ============================================
# TASK 2: Generar Reporte de Clustering
# ============================================
# Opcional: Generar reporte específico de clustering
generate_clustering_report = BashOperator(
    task_id='generate_clustering_report',
    bash_command='docker exec -w /app ml-letterboxd-pipeline python -c "import pandas as pd; import json; df = pd.read_csv(\'data/07_model_output/clustering_comparison.csv\'); print(\'\\n=== COMPARACIÓN DE MODELOS DE CLUSTERING ===\\n\'); print(df.to_string(index=False)); metrics = json.load(open(\'data/07_model_output/clustering_metrics.json\')); print(\'\\n=== MÉTRICAS DETALLADAS ===\\n\'); print(json.dumps(metrics, indent=2))"',
    dag=dag,
    execution_timeout=timedelta(minutes=10),  # Timeout de 10 minutos para el reporte
)

# ============================================
# DEFINIENDO DEPENDENCIAS
# ============================================
# Orden de ejecución:
# 1. Ejecutar pipeline completo de unsupervised learning
# 2. Generar reporte de clustering

unsupervised_learning >> generate_clustering_report


