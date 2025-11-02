# 🚀 Guía Completa de Ejecución del Proyecto

## 📋 Índice
1. [¿Qué es DVC y para qué sirve?](#qué-es-dvc)
2. [Requisitos Previos](#requisitos-previos)
3. [Paso 1: Preparar el Entorno](#paso-1-preparar-el-entorno)
4. [Paso 2: Ejecutar con Docker y Airflow](#paso-2-ejecutar-con-docker-y-airflow)
5. [Paso 3: Ver Resultados en Airflow](#paso-3-ver-resultados-en-airflow)
6. [Paso 4: Usar DVC para Versionado](#paso-4-usar-dvc-para-versionado)
7. [Paso 5: Ver Resultados Finales](#paso-5-ver-resultados-finales)
8. [Troubleshooting](#troubleshooting)

---

## 🔍 ¿Qué es DVC y para qué sirve?

**DVC (Data Version Control)** es como Git pero para datos y modelos.

### ¿Por qué usar DVC?
- ✅ **Versionado de datos**: Guarda versiones de datasets grandes sin ocupar espacio en Git
- ✅ **Reproducibilidad**: Puedes reproducir exactamente los mismos resultados
- ✅ **Métricas versionadas**: Guarda las métricas de cada experimento
- ✅ **Colaboración**: Compartir datasets y modelos sin problemas de tamaño

### ¿Cómo funciona?
1. DVC guarda los datos en almacenamiento remoto (o local)
2. Git solo guarda referencias pequeñas (archivos `.dvc`)
3. Cuando alguien clona el proyecto, puede descargar los datos con `dvc pull`

**En este proyecto, DVC versiona:**
- Datasets procesados (`data/02_intermediate/`, `data/03_primary/`)
- Datasets para ML (`data/05_model_input/`)
- Modelos entrenados (`data/06_models/`)
- Métricas de evaluación (JSON con resultados)

---

## 📦 Requisitos Previos

### Software necesario:
- ✅ **Docker Desktop** instalado y ejecutándose
- ✅ **Git** (para versionado de código)
- ✅ **DVC** (opcional, para versionado de datos)

### Verificar instalaciones:
```bash
# Verificar Docker
docker --version
docker-compose --version

# Verificar Git
git --version

# Verificar DVC (si ya está instalado)
dvc --version
```

### 📦 Instalación de DVC

**Tienes 3 opciones:**

#### Opción 1: Instalación Global (Recomendada para empezar) ⭐
```bash
# Instalar DVC globalmente en tu computadora
pip install dvc

# Verificar instalación
dvc --version
```
✅ **Ventajas**: Simple, disponible en cualquier proyecto  
⚠️ **Desventajas**: Puede causar conflictos de versiones entre proyectos

#### Opción 2: Instalación en el Proyecto (Mejor práctica)
```bash
# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar DVC solo en este proyecto
pip install dvc

# O agregar a requirements.txt y luego:
pip install -r requirements.txt
```
✅ **Ventajas**: Aísla dependencias, no afecta otros proyectos  
✅ **Mejor para**: Proyectos que compartirás o producción

#### Opción 3: Solo usar DVC en Docker (No instalar localmente)
Si solo ejecutas con Docker, **NO necesitas instalar DVC localmente**.  
El Dockerfile ya puede incluirlo si lo agregas a `requirements.txt`.

---

## 🛠️ Paso 1: Preparar el Entorno

### 1.1. Navegar al directorio del proyecto
```bash
cd "C:\Users\mathi\OneDrive\Escritorio\Proyecto kedro\ev1MachineL"
```

### 1.2. Verificar que los datos existen
```bash
# Verificar que los datos raw están presentes
dir data\01_raw\
```

Debes ver:
- `movies.csv`
- `releases.csv`
- `genres.csv`
- `countries.csv`

### 1.3. (Opcional) Inicializar DVC si es la primera vez
```bash
# Solo si es la primera vez que usas DVC en este proyecto
dvc init

# Configurar almacenamiento remoto (opcional)
# dvc remote add -d myremote /path/to/storage
```

---

## 🐳 Paso 2: Ejecutar con Docker y Airflow

### 2.1. Construir la imagen Docker (solo la primera vez o después de cambios)
```bash
# Construir imagen del proyecto
docker build -t kedro-ml .
```

⏱️ **Tiempo estimado**: 5-10 minutos (solo la primera vez)

### 2.2. Iniciar todos los servicios (Docker Compose)
```bash
# Iniciar Airflow + Postgres + Pipeline Kedro
docker-compose up -d
```

Esto iniciará:
- 🗄️ **Postgres**: Base de datos para Airflow
- 🔄 **Airflow-init**: Inicializa Airflow (solo una vez)
- 🌐 **Airflow-webserver**: Interfaz web (puerto 8080)
- ⚙️ **Airflow-scheduler**: Ejecuta los DAGs (IMPORTANTE: sin esto los DAGs no se ejecutan)
- 🚀 **Kedro-pipeline**: Ejecuta el pipeline automáticamente

⏱️ **Tiempo estimado**: 2-3 minutos para iniciar

### 2.3. Verificar que todo está corriendo
```bash
# Ver estado de los contenedores
docker-compose ps
```

Debes ver todos los servicios como **Up**:
```
NAME                   STATUS
airflow-init           Exited (0)
airflow-webserver      Up
airflow-scheduler      Up  ⬅️ IMPORTANTE: Sin esto los DAGs no ejecutan
postgres               Up
ml-letterboxd-pipeline Up
```

**⚠️ Si no ves `airflow-scheduler` en la lista, los DAGs NO se ejecutarán aunque los actives.**

### 2.4. Ver logs (opcional)

**⚠️ Importante**: El comando `logs -f` NO termina solo. Muestra logs en tiempo real hasta que presiones `Ctrl + C`.

```bash
# Ver logs que TERMINAN (últimos 100 líneas)
docker-compose logs --tail=100

# Ver logs que NO TERMINAN (siguen mostrando nuevos logs)
docker-compose logs -f
# Presiona Ctrl + C para detener

# Ver logs solo de Airflow
docker-compose logs -f airflow-webserver
# Presiona Ctrl + C para detener

# Ver logs del pipeline Kedro (últimos 50)
docker-compose logs --tail=50 kedro-pipeline

# Ver logs en tiempo real del pipeline
docker-compose logs -f kedro-pipeline
# Presiona Ctrl + C para detener
```

**💡 Tip**: Usa `--tail=N` si quieres ver logs y que termine automáticamente. Usa `-f` si quieres monitorear en tiempo real.

---

## 🌐 Paso 3: Ver Resultados en Airflow

### 3.1. Acceder a la interfaz de Airflow

1. **Abrir navegador** y ir a: `http://localhost:8080`

2. **Iniciar sesión**:
   - Usuario: `admin`
   - Contraseña: `admin`

### 3.2. Ver los DAGs disponibles

En la interfaz de Airflow verás los siguientes DAGs:

1. **`kedro_ml_pipeline`**: Pipeline completo (preparación + clasificación + regresión)
2. **`kedro_classification_pipeline`**: Solo clasificación
3. **`kedro_regression_pipeline`**: Solo regresión

### 3.3. Activar y ejecutar un DAG

1. **Buscar el DAG** `kedro_ml_pipeline` en la lista
2. **Activar el DAG**: Click en el toggle a la izquierda del nombre
3. **Ejecutar manualmente**: Click en el botón ▶️ "Trigger DAG"

### 3.4. Monitorear la ejecución

1. **Ver el estado**: Los círculos de colores indican el estado:
   - 🟢 Verde = Completado exitosamente
   - 🟡 Amarillo = En ejecución
   - 🔴 Rojo = Error
   - ⚪ Gris = No ejecutado aún

2. **Ver detalles de cada tarea**:
   - Click en el círculo de una tarea
   - Verás: Logs, detalles, duración, etc.

3. **Ver logs completos**:
   - Click en una tarea → Click en "Log"
   - Verás todos los mensajes de ejecución

### 3.5. Entender el flujo del DAG

El DAG `kedro_ml_pipeline` ejecuta en este orden:

```
1. prepare_data       → Limpia y prepara datos raw
2. create_features    → Genera features para ML
3. train_classification → Entrena 5 modelos de clasificación
4. train_regression   → Entrena 5 modelos de regresión
5. evaluate_models    → Compara y selecciona mejores modelos
```

---

## 📊 Paso 4: Usar DVC para Versionado

### 4.1. ¿Qué versiona DVC en este proyecto?

DVC rastrea automáticamente (según `dvc.yaml`):

**Datos procesados:**
- `data/02_intermediate/*.csv` (datos limpios)
- `data/03_primary/*.csv` (datos integrados)
- `data/05_model_input/*.csv` (datasets para ML)

**Modelos y métricas:**
- `data/06_models/classification_results.pkl`
- `data/06_models/regression_results.pkl`
- `data/06_models/*_metrics.json`

**Visualizaciones:**
- `data/08_reporting/*.png` (gráficos generados)

### 4.2. Ejecutar pipeline completo con DVC

```bash
# Reproducir todo el pipeline (preparación → evaluación)
dvc repro

# Ver qué se ejecutó
dvc dag
```

### 4.3. Ver métricas versionadas

```bash
# Ver todas las métricas guardadas
dvc metrics show

# Ver métricas específicas
dvc metrics show data/06_models/classification_metrics.json
dvc metrics show data/06_models/regression_metrics.json

# Comparar métricas entre commits
dvc metrics diff
```

### 4.4. Guardar cambios en DVC

```bash
# Después de ejecutar el pipeline, guardar en DVC
dvc add data/06_models/classification_metrics.json
dvc add data/06_models/regression_metrics.json

# Commit en Git (DVC crea archivos .dvc que se versionan en Git)
git add data/06_models/*.dvc .dvc/.gitignore
git commit -m "Actualizar métricas de modelos"
```

### 4.5. Reproducir un stage específico

```bash
# Solo preparar datos
dvc repro prepare

# Solo entrenar clasificación
dvc repro train_classification

# Solo entrenar regresión
dvc repro train_regression
```

---

## 📁 Paso 5: Ver Resultados Finales

### 5.1. Métricas de Modelos

**Clasificación:**
```bash
# Ver métricas (desde terminal)
type data\06_models\classification_metrics.json

# O abrir en navegador/editor
notepad data\06_models\classification_metrics.json
```

**Regresión:**
```bash
type data\06_models\regression_metrics.json
```

### 5.2. Visualizaciones Generadas

Ver gráficos en:
```
data/08_reporting/
├── fase5_classification_comparison.png  (Comparación modelos clasificación)
├── fase5_regression_comparison.png      (Comparación modelos regresión)
├── cleaning_process.png                 (Proceso de limpieza)
├── comparative_analysis.png             (Análisis comparativo)
├── genre_analysis.png                   (Análisis de géneros)
└── temporal_analysis.png                (Análisis temporal)
```

### 5.3. Reportes Finales

```bash
# Reporte de evaluación (Fase 5)
type data\06_models\fase5_evaluation_report.json

# Reporte de comparación (si existe)
type data\07_model_output\comparison_report.md
```

### 5.4. Ejecutar Notebooks para Análisis Detallado

```bash
# Abrir Jupyter (desde el contenedor o localmente)
jupyter notebook notebooks/

# Ejecutar en orden:
# 1. Fase1.ipynb - Comprensión del negocio
# 2. Fase2.ipynb - Exploración de datos
# 3. Fase3.ipynb - Preparación de datos
# 4. Fase4_Clasificacion.ipynb - Modelos de clasificación
# 5. Fase4_Regresion.ipynb - Modelos de regresión
# 6. Fase5_Evaluacion.ipynb - Evaluación y selección
# 7. Fase6_Despliegue.ipynb - Conclusiones
```

---

## 🔄 Flujo Completo Recomendado

### Opción A: Ejecución Rápida (Primera vez)

```bash
# 1. Construir imagen
docker build -t kedro-ml .

# 2. Iniciar servicios
docker-compose up -d

# 3. Esperar 2-3 minutos y abrir Airflow
# http://localhost:8080
# Usuario: admin / Password: admin

# 4. Activar DAG "kedro_ml_pipeline"
# 5. Ejecutar manualmente
# 6. Ver resultados en data/06_models/
```

### Opción B: Con DVC (Versionado Completo)

```bash
# 1. Construir e iniciar (igual que Opción A)
docker build -t kedro-ml .
docker-compose up -d

# 2. Esperar a que termine la ejecución automática
# 3. Verificar que se generaron los archivos

# 4. Versionar con DVC
dvc add data/06_models/classification_metrics.json
dvc add data/06_models/regression_metrics.json

# 5. Commit en Git
git add data/06_models/*.dvc .dvc/.gitignore
git commit -m "Guardar métricas de modelos"

# 6. Ver métricas
dvc metrics show
```

---

## 🛑 Detener Servicios

```bash
# Detener todos los servicios
docker-compose down

# Detener y eliminar volúmenes (⚠️ borra datos de Postgres)
docker-compose down -v

# Detener solo un servicio
docker-compose stop airflow-webserver
```

---

## 🔧 Troubleshooting

### ⚠️ Comando `logs -f` no termina
Si ejecutaste `docker-compose logs -f` y sigue mostrando logs:
- **Es normal**: El `-f` significa "follow" (seguir mostrando nuevos logs)
- **Para detener**: Presiona `Ctrl + C`
- **Para ver logs que terminan**: Usa `docker-compose logs --tail=100` (sin `-f`)

### Problema: Airflow no inicia
```bash
# Ver logs de errores (que terminan)
docker-compose logs --tail=100 airflow-webserver

# Ver logs en tiempo real (presiona Ctrl+C para detener)
docker-compose logs -f airflow-webserver

# Reiniciar servicios
docker-compose restart

# Reconstruir desde cero
docker-compose down -v
docker-compose up -d --build
```

### Problema: DAG activado pero no se ejecuta (nada pasa al presionar play) O DAG ejecuta pero falla

**🔴 PROBLEMA COMÚN**: Falta el Airflow Scheduler.

**Síntomas**:
- DAGs aparecen en la interfaz
- Puedes activarlos (toggle ON)
- Pero al presionar "Trigger DAG" no pasa nada

**Solución**:
1. Verificar que el scheduler esté corriendo:
   ```bash
   docker ps --filter "name=scheduler"
   ```
   
2. Si no aparece, agregar scheduler al `docker-compose.yml` (ya está incluido en la versión actualizada)

3. Reiniciar servicios:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

4. Verificar que scheduler esté activo:
   ```bash
   docker-compose ps
   ```
   
   Debes ver: `airflow-scheduler` con estado **Up**

5. Esperar 1-2 minutos para que el scheduler detecte los DAGs

### Problema: DAG no aparece en Airflow
- Verificar que el archivo está en `dags/`
- Verificar que no tiene errores de sintaxis
- Esperar 30-60 segundos (Airflow tarda en cargar DAGs)

### Problema: Puerto 8080 ocupado
```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8081:8080"  # Usar puerto 8081 en lugar de 8080
```

### Problema: Contenedor se detiene
```bash
# Ver logs del contenedor
docker logs ml-letterboxd-pipeline

# Ejecutar manualmente dentro del contenedor
docker exec -it ml-letterboxd-pipeline bash
kedro run
```

### Problema: DVC no encuentra archivos
```bash
# Verificar que los archivos existen
ls -la data/06_models/

# Reproducir el stage que genera el archivo
dvc repro train_classification  # Para classification_metrics.json
```

### 🔍 Cómo Revisar Errores en Airflow

Cuando un DAG falla (círculo rojo en Airflow UI):

#### **Paso 1: Ver el Error en la UI**

1. **Click en el DAG** que falló (ej: `kedro_classification`)
2. **Click en el círculo de color** de la tarea que falló (rojo = fallido)
3. **Click en "Log"** en el menú que aparece
4. **Ver el error completo** en el log

#### **Paso 2: Verificar el Contenedor de Kedro**

```bash
# Verificar que el contenedor está corriendo
docker ps | Select-String "ml-letterboxd-pipeline"

# Si no está corriendo, iniciarlo
docker-compose up -d kedro-pipeline

# Probar el comando manualmente
docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=classification_pipeline
```

#### **Paso 3: Ver Logs desde Terminal**

```bash
# Ver logs del scheduler (últimos 100 líneas)
docker-compose logs airflow-scheduler --tail 100

# Ver logs de una tarea específica
docker-compose exec airflow-webserver ls -la /opt/airflow/logs/dag_id=kedro_classification/

# Ver logs del contenedor de Kedro
docker logs ml-letterboxd-pipeline --tail 50
```

#### **Paso 4: Errores Comunes y Soluciones**

**Error: "docker exec: No such container"**
- **Solución**: El contenedor de Kedro no está corriendo
  ```bash
  docker-compose up -d kedro-pipeline
  ```

**Error: "cd /app: No such file or directory"**
- **Solución**: Ya corregido. Los DAGs ahora usan `docker exec -w /app`. Si aparece, recarga los DAGs:
  ```bash
  docker-compose restart airflow-scheduler
  ```

**Error: "Pipeline 'classification_pipeline' not found"**
- **Solución**: Verificar que el pipeline existe en `src/letterboxdml/pipelines/ml_modeling_pipeline.py`
  ```bash
  docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=eda_pipeline
  ```

**Error: "No module named 'X'"**
- **Solución**: Instalar dependencia faltante en el contenedor
  ```bash
  docker exec -w /app ml-letterboxd-pipeline pip install <nombre_modulo>
  ```

#### **Paso 5: Verificar Estado Completo**

```bash
# Estado de todos los contenedores
docker-compose ps

# Todos deben estar "Up" y "healthy":
# ✅ airflow-webserver - Up (healthy)
# ✅ airflow-scheduler - Up (healthy)
# ✅ ml-letterboxd-pipeline - Up (healthy)
# ✅ postgres - Up (healthy)
```

---

## 📝 Checklist Final

Después de ejecutar todo, verifica:

- [ ] ✅ Airflow accesible en http://localhost:8080
- [ ] ✅ DAGs visibles en Airflow
- [ ] ✅ Pipeline ejecutado exitosamente (círculos verdes)
- [ ] ✅ Archivos generados en `data/06_models/`:
  - [ ] `classification_metrics.json`
  - [ ] `regression_metrics.json`
  - [ ] `fase5_evaluation_report.json`
- [ ] ✅ Visualizaciones en `data/08_reporting/`
- [ ] ✅ (Opcional) Métricas versionadas en DVC

---

## 📚 Recursos Adicionales

- **Kedro Docs**: https://kedro.readthedocs.io
- **Airflow Docs**: https://airflow.apache.org/docs
- **DVC Docs**: https://dvc.org/doc

---

**Autores**: Mathias Jara & Eduardo Gonzalez

