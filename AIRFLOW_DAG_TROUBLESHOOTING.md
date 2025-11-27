# 🔧 Troubleshooting: DAG de Unsupervised Learning no aparece en Airflow

## ✅ Solución Aplicada

Se ha creado el archivo `dags/kedro_unsupervised_learning_dag.py` con el DAG dedicado para el pipeline de aprendizaje no supervisado.

## 📋 Pasos para Verificar

### 1. Verificar que el archivo existe
```bash
ls dags/kedro_unsupervised_learning_dag.py
```

### 2. Verificar que el volumen está montado
El archivo `docker-compose.yml` debe tener el volumen montado:
```yaml
volumes:
  - ./dags:/opt/airflow/dags
```

### 3. Reiniciar Airflow Scheduler
```bash
docker-compose restart airflow-scheduler
```

O reiniciar todos los servicios:
```bash
docker-compose restart
```

### 4. Verificar logs del scheduler
```bash
docker-compose logs airflow-scheduler | Select-String -Pattern "unsupervised"
```

### 5. Verificar que el DAG se carga sin errores
En la UI de Airflow:
1. Ir a la página de DAGs
2. Buscar "kedro_unsupervised_learning"
3. Si aparece en rojo, hacer clic para ver los errores

### 6. Forzar recarga de DAGs
En la UI de Airflow:
1. Ir a "Admin" → "DAGs"
2. Hacer clic en el botón de refresh (🔄) o presionar F5

## 🐛 Problemas Comunes

### Problema 1: El DAG no aparece en la lista
**Causa**: El scheduler no ha detectado el nuevo archivo
**Solución**: 
- Esperar 30-60 segundos (Airflow escanea cada 30 segundos)
- Reiniciar el scheduler: `docker-compose restart airflow-scheduler`

### Problema 2: El DAG aparece en rojo (con error)
**Causa**: Error de sintaxis o importación en el DAG
**Solución**:
- Verificar logs: `docker-compose logs airflow-scheduler`
- Verificar sintaxis: `python -m py_compile dags/kedro_unsupervised_learning_dag.py`

### Problema 3: El contenedor ml-letterboxd-pipeline no existe
**Causa**: El contenedor no está corriendo
**Solución**:
```bash
docker-compose up -d kedro-pipeline
```

### Problema 4: Permisos de archivo
**Causa**: El archivo no tiene permisos de lectura
**Solución**:
```bash
chmod 644 dags/kedro_unsupervised_learning_dag.py
```

## ✅ Verificación Final

Después de seguir los pasos anteriores, deberías ver en Airflow UI:

1. **4 DAGs en total**:
   - `kedro_classification`
   - `kedro_regression`
   - `kedro_ml_pipeline`
   - `kedro_unsupervised_learning` ⭐ **NUEVO**

2. **Tags del nuevo DAG**:
   - `kedro`
   - `machine-learning`
   - `unsupervised`
   - `clustering`
   - `dimensionality-reduction`

3. **Tasks del nuevo DAG**:
   - `run_unsupervised_learning_pipeline`
   - `generate_clustering_report`

## 🚀 Ejecutar el DAG

1. En Airflow UI, encontrar `kedro_unsupervised_learning`
2. Activar el toggle (si está pausado)
3. Hacer clic en el botón "Play" ▶️
4. Seleccionar "Trigger DAG"

## 📝 Notas

- El DAG está configurado con `schedule_interval=None`, por lo que solo se ejecuta manualmente
- El DAG requiere que el contenedor `ml-letterboxd-pipeline` esté corriendo
- El pipeline necesita que `regression_dataset.csv` exista (generado por `eda_pipeline`)

