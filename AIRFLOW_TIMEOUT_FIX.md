# 🔧 Solución: Pipelines Fallan en Airflow por Timeout

## ❌ Problema Identificado

Todos los pipelines estaban fallando en Airflow porque las tareas excedían el **timeout por defecto** de Airflow (aproximadamente 5-10 minutos). Los pipelines de Kedro, especialmente el de **unsupervised learning**, pueden tardar mucho tiempo:

- **Clustering**: Búsqueda de k óptimo puede tardar 10-15 minutos
- **Reducción Dimensional**: t-SNE y UMAP pueden tardar 20-30 minutos cada uno
- **Entrenamiento de Modelos**: Puede tardar 30-60 minutos dependiendo del tamaño del dataset

## ✅ Solución Aplicada

Se agregaron **timeouts explícitos** a todas las tareas de los DAGs de Airflow:

### Timeouts Configurados

| Pipeline/Tarea | Timeout | Razón |
|----------------|---------|-------|
| `data_preparation_pipeline` | 1 hora | Preparación de datos puede ser extensa |
| `eda_pipeline` | 1 hora | Análisis exploratorio completo |
| `classification_pipeline` | 2 horas | Entrenamiento de múltiples modelos |
| `regression_pipeline` | 2 horas | Entrenamiento de múltiples modelos |
| `unsupervised_learning_pipeline` | 2 horas | Clustering + Reducción dimensional (muy intensivo) |
| `ml_modeling_pipeline` | 1 hora | Evaluación de modelos |
| Tareas de evaluación/reporte | 10-30 minutos | Operaciones rápidas |

### Archivos Modificados

1. `dags/kedro_ml_dag.py` - DAG principal con todos los pipelines
2. `dags/kedro_unsupervised_learning_dag.py` - DAG de aprendizaje no supervisado
3. `dags/kedro_classification_dag.py` - DAG de clasificación
4. `dags/kedro_regression_dag.py` - DAG de regresión

### Ejemplo de Cambio

**Antes:**
```python
unsupervised_learning = BashOperator(
    task_id='run_unsupervised_learning_pipeline',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=unsupervised_learning_pipeline',
    dag=dag,
)
```

**Después:**
```python
unsupervised_learning = BashOperator(
    task_id='run_unsupervised_learning_pipeline',
    bash_command='docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=unsupervised_learning_pipeline',
    dag=dag,
    execution_timeout=timedelta(hours=2),  # Timeout de 2 horas
)
```

## 🚀 Próximos Pasos

1. **Reiniciar Airflow Scheduler**:
   ```bash
   docker-compose restart airflow-scheduler
   ```

2. **Verificar que los DAGs se cargaron correctamente**:
   - Ir a http://localhost:8080
   - Verificar que los DAGs aparecen sin errores (no en rojo)

3. **Ejecutar un pipeline de prueba**:
   - Activar el DAG `kedro_unsupervised_learning`
   - Trigger manual del DAG
   - Monitorear que la tarea no falle por timeout

## 📊 Monitoreo

Para monitorear el tiempo de ejecución de las tareas:

1. En Airflow UI, ir a la tarea específica
2. Ver "Duration" en el gráfico de ejecución
3. Si una tarea se acerca al timeout, considerar:
   - Aumentar el timeout si es necesario
   - Optimizar el pipeline de Kedro para reducir tiempo de ejecución
   - Dividir el pipeline en tareas más pequeñas

## ⚠️ Notas Importantes

- Los timeouts son **máximos**, no objetivos. Las tareas deberían completarse antes del timeout.
- Si una tarea falla por timeout, Airflow la marcará como "failed" y puede reintentarla según la configuración de `retries`.
- El timeout se cuenta desde el inicio de la ejecución de la tarea, no desde el inicio del DAG.

## 🔍 Verificación

Para verificar que los cambios funcionaron:

```bash
# Ver logs del scheduler
docker-compose logs airflow-scheduler --tail 50

# Probar ejecutar un comando manualmente
docker exec airflow-scheduler docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=unsupervised_learning_pipeline
```

Si el comando funciona manualmente pero falla en Airflow, el problema puede ser:
- Timeout aún insuficiente (aumentar más)
- Problemas de recursos (CPU/RAM)
- Problemas de red entre contenedores

