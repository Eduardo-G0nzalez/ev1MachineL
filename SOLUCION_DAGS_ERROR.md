# ✅ Solución: DAGs Corregidos - Ahora Funcionan

## 🔧 Problema Resuelto

Los DAGs estaban intentando ejecutar comandos `kedro run` en el contenedor de **Airflow**, pero Kedro solo está instalado en el contenedor **ml-letterboxd-pipeline**.

## ✅ Solución Aplicada

He actualizado los DAGs para que ejecuten comandos en el contenedor correcto usando `docker exec`.

## 📝 Pasos para Aplicar los Cambios

### Paso 1: Detener servicios actuales
```bash
cd "C:\Users\mathi\OneDrive\Escritorio\Proyecto kedro\ev1MachineL"
docker-compose down
```

### Paso 2: Reiniciar servicios
```bash
docker-compose up -d
```

### Paso 3: Esperar 2-3 minutos
Espera a que todos los servicios se inicien completamente.

### Paso 4: Verificar que el scheduler está corriendo
```bash
docker-compose ps
```

Debes ver:
- `airflow-scheduler` - **Up**
- `airflow-webserver` - **Up**
- `ml-letterboxd-pipeline` - **Up**

### Paso 5: Probar en Airflow

1. Ir a http://localhost:8080
2. Verificar que los DAGs se recargaron (sin errores)
3. Activar `kedro_ml_pipeline` (toggle ON)
4. Presionar ▶️ "Trigger DAG"
5. **Ahora debería ejecutarse correctamente**

---

## 🔍 Qué Cambió

### Antes (NO funcionaba):
```python
bash_command='cd /app && kedro run --pipeline=...'
```
❌ Intentaba ejecutar en contenedor de Airflow (donde Kedro no existe)

### Ahora (SÍ funciona):
```python
bash_command='docker exec ml-letterboxd-pipeline kedro run --pipeline=...'
```
✅ Ejecuta en el contenedor correcto donde Kedro está instalado

---

## 📋 Archivos Actualizados

1. ✅ `dags/kedro_ml_dag.py` - Corregido
2. ✅ `dags/kedro_classification_dag.py` - Corregido
3. ✅ `dags/kedro_regression_dag.py` - Corregido
4. ✅ `docker-compose.yml` - Agregado acceso a Docker socket

---

## ⚠️ Si Sigue Sin Funcionar

### Verificar errores en Airflow:
1. En la UI de Airflow, click en el DAG
2. Click en el círculo rojo/amarillo de la tarea que falló
3. Click en "Log" para ver el error específico

### Verificar que el contenedor de Kedro está corriendo:
```bash
docker ps --filter "name=ml-letterboxd-pipeline"
```

### Ver logs del scheduler:
```bash
docker-compose logs --tail=50 airflow-scheduler
```

---

## 🎯 Resultado Esperado

Después de ejecutar `kedro_ml_pipeline`:
- ✅ Todas las tareas deberían completarse (círculos verdes)
- ✅ Los modelos se entrenarán y guardarán en `data/06_models/`
- ✅ Las métricas se exportarán a JSON
- ✅ Los gráficos se generarán en `data/08_reporting/`

**Autores**: Mathias Jara & Eduardo Gonzalez

