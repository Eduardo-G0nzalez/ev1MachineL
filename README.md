# 🎬 Letterboxd Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Kedro](https://img.shields.io/badge/Kedro-0.18+-green.svg)](https://kedro.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Airflow](https://img.shields.io/badge/Airflow-2.7.0-orange.svg)](https://airflow.apache.org)
[![DVC](https://img.shields.io/badge/DVC-3.0+-yellow.svg)](https://dvc.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📷 Video explicativo (evaluación 1)
https://drive.google.com/file/d/1As-mk4WYkrqv6CfL4AItxU33odSIaEzI/view?usp=sharing

## 📋 Descripción del Proyecto

Este proyecto de **Machine Learning** implementa metodología **CRISP-DM** completa para analizar datos cinematográficos de **Letterboxd**. Utiliza **Kedro**, **Docker**, **Airflow** y **DVC** para crear un pipeline robusto y reproducible que estudia la evolución de géneros cinematográficos entre las décadas de 2000s y 2010s en Estados Unidos.

### 🎯 Objetivos del Proyecto

- ✅ **Análisis Exploratorio de Datos (EDA)**: Comprensión profunda de estructura y calidad
- ✅ **Preparación de Datos**: Limpieza, transformación e integración de datasets
- ✅ **Modelado de Machine Learning**: Clasificación y regresión con ≥5 modelos cada uno
- ✅ **Evaluación y Selección**: Comparación de modelos y selección de mejores
- ✅ **Despliegue**: Plan de producción con monitoreo

### 📊 Datasets

- **Fuente**: https://www.kaggle.com/datasets/gsimonx37/letterboxd
- **movies.csv**: 941,597 películas
- **releases.csv**: 1,332,782 estrenos
- **countries.csv**: 693,476 países
- **genres.csv**: 1,046,849 géneros

---

## 🚀 Inicio Rápido

> 📖 **Para una guía completa paso a paso, ver**: [`GUIA_EJECUCION_COMPLETA.md`](GUIA_EJECUCION_COMPLETA.md)

### Opción 1: Docker + Airflow (Recomendado) ⭐

```bash
# 1. Clonar repositorio
git clone https://github.com/Eduardo-G0nzalez/ev1MachineL.git
cd ev1MachineL

# 2. Iniciar todos los servicios (Docker + Airflow + Postgres)
docker-compose up -d

# 3. Acceder a Airflow UI
# http://localhost:8080
# Usuario: admin
# Password: admin

# 4. Activar y ejecutar el DAG: kedro_ml_pipeline
```

### Opción 2: Ejecución Local con Kedro

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo
kedro run

# Pipeline específico
kedro run --pipeline=classification_pipeline
kedro run --pipeline=regression_pipeline
```

### Opción 3: Jupyter Notebooks

```bash
cd notebooks
jupyter notebook

# Ejecutar notebooks en orden:
# Fase1.ipynb → Fase2.ipynb → ... → Fase6.ipynb
```

---

## 📚 Fases del Proyecto (CRISP-DM)

### 🔍 Fase 1: Comprensión del Negocio
- **Notebook**: `Fase1.ipynb`
- Análisis inicial y definición de objetivos ML
- Identificación de hipótesis de negocio

### 📊 Fase 2: Comprensión de Datos (EDA)
- **Notebook**: `Fase2.ipynb`
- Análisis de calidad: valores faltantes, outliers, completitud
- Análisis temporal: tendencias, estacionalidad
- Visualizaciones avanzadas (12+ gráficos)

### 🧹 Fase 3: Preparación de Datos
- **Notebook**: `Fase3.ipynb`
- Limpieza: normalización, deduplicación, filtrado
- Transformación: creación de variables, integración
- Análisis comparativo: 2000s vs 2010s

### 🤖 Fase 4: Modelado
- **Clasificación**: `Fase4_Clasificacion.ipynb`
  - Target: Éxito comercial (Alto/Medio/Bajo)
  - Modelos: Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost
  - Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  
- **Regresión**: `Fase4_Regresion.ipynb`
  - Target: Rating de audiencia (0-5)
  - Modelos: Linear Regression, Random Forest, Gradient Boosting, KNN, XGBoost
  - Métricas: R², RMSE, MAE, MSE

### 📈 Fase 5: Evaluación
- **Notebook**: `Fase5_Evaluacion.ipynb`
- Comparación de modelos
- Selección de mejores modelos
- Validación contra objetivos de negocio

### 🚀 Fase 6: Despliegue
- **Notebook**: `Fase6_Despliegue.ipynb`
- Plan de infraestructura
- Estrategias de monitoreo
- Limitaciones y mejoras futuras

---

## 🏗️ Arquitectura del Sistema

### Componentes Principales

- **Kedro**: Framework de data engineering y pipelines modulares
- **DVC**: Versionado de datos, modelos y métricas
- **Airflow**: Orquestación automatizada de workflows (DAGs)
- **Docker**: Contenerización para reproducibilidad
- **scikit-learn**: Machine Learning con GridSearchCV (k=5)

### Flujo de Datos

```
┌─────────────┐
│ Datos Raw  │
└──────┬──────┘
       │
       v
┌─────────────────┐
│ Pipeline Kedro  │ ← Docker Container
│ (Preparación)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ DVC Stages      │ ← Versionado
│ (Reproducible)  │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Airflow DAGs    │ ← Orquestación
│ (kedro_ml_pipeline)│
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Modelos +       │
│ Métricas JSON   │
└─────────────────┘
```

### Pipelines Kedro Disponibles

- `data_preparation_pipeline`: Limpieza e integración
- `eda_pipeline`: Análisis exploratorio
- `classification_pipeline`: Modelos de clasificación
- `regression_pipeline`: Modelos de regresión
- `ml_modeling_pipeline`: Pipeline combinado

### DAGs de Airflow

- `kedro_ml_pipeline`: Pipeline completo (preparación → ML → evaluación)
- `kedro_classification`: Solo clasificación
- `kedro_regression`: Solo regresión

---

## 📁 Estructura del Proyecto

```
ev1MachineL/
├── notebooks/              # Notebooks Fase 1-6 (CRISP-DM completo)
│   ├── Fase1.ipynb
│   ├── Fase2.ipynb
│   ├── Fase3.ipynb
│   ├── Fase4_Clasificacion.ipynb
│   ├── Fase4_Regresion.ipynb
│   ├── Fase5_Evaluacion.ipynb
│   └── Fase6_Despliegue.ipynb
├── data/
│   ├── 01_raw/            # Datos originales
│   ├── 02_intermediate/    # Datos procesados (versionados con DVC)
│   ├── 03_primary/         # Datos finales (versionados con DVC)
│   ├── 05_model_input/     # Datos para ML (versionados con DVC)
│   ├── 06_models/          # Modelos y métricas (versionados con DVC)
│   └── 08_reporting/       # Gráficos y visualizaciones
├── src/letterboxdml/
│   └── pipelines/          # Pipelines Kedro modulares
├── dags/                   # DAGs de Airflow
│   ├── kedro_ml_dag.py
│   ├── kedro_classification_dag.py
│   └── kedro_regression_dag.py
├── dvc.yaml                # Configuración DVC (versionado)
├── docker-compose.yml      # Orquestación Docker
├── Dockerfile              # Imagen Docker
└── requirements.txt        # Dependencias Python
```

---

## ✅ Requisitos Implementados

- ✅ **Metodología CRISP-DM** completa (Fases 1-6)
- ✅ **Pipelines Kedro** modulares y ejecutables
- ✅ **DVC** para versionado de datos, features y modelos
- ✅ **Airflow** con DAGs orquestados
- ✅ **Docker** con imagen reproducible
- ✅ **≥5 modelos** por tipo (clasificación y regresión)
- ✅ **GridSearchCV** con validación cruzada (k=5)
- ✅ **Métricas completas** con mean±std
- ✅ **Tabla comparativa** de resultados
- ✅ **Evaluación y selección** de mejores modelos
- ✅ **Plan de despliegue** y monitoreo
- ✅ **Documentación técnica** completa

---

## 🎯 Resultados Principales

### Clasificación (Éxito Comercial)
- **Mejor modelo**: XGBoost
- **Accuracy**: ~75%
- **Métricas completas**: Ver `data/06_models/classification_metrics.json`

### Regresión (Rating de Audiencia)
- **Mejor modelo**: Random Forest / Gradient Boosting
- **R² Score**: ~40-45%
- **Métricas completas**: Ver `data/06_models/regression_metrics.json`

> 📊 **Nota**: Métricas completas disponibles después de ejecutar el pipeline completo.

---

## 🔧 Tecnologías Utilizadas

- **Python 3.8+**
- **Kedro** - Framework de data engineering
- **scikit-learn** - Machine Learning
- **Pandas / NumPy** - Manipulación de datos
- **Matplotlib / Seaborn** - Visualizaciones
- **Jupyter** - Notebooks interactivos
- **Docker** - Contenerización
- **Docker Compose** - Orquestación multi-container
- **Apache Airflow** - Workflow orchestration
- **DVC** - Data version control
- **PostgreSQL** - Base de datos para Airflow

---

## 📖 Documentación Adicional

- [`GUIA_EJECUCION_COMPLETA.md`](GUIA_EJECUCION_COMPLETA.md) - Guía paso a paso completa
- [`QUICK_START.md`](QUICK_START.md) - Inicio rápido
- [`GUIA_DVC_GITHUB.md`](GUIA_DVC_GITHUB.md) - Cómo usar DVC con GitHub
- [`INSTALACION_DVC.md`](INSTALACION_DVC.md) - Instalación de DVC
- [`SOLUCION_AIRFLOW.md`](SOLUCION_AIRFLOW.md) - Troubleshooting Airflow

---

## 👨‍💻 Autores

**Mathias Jara** - Full Stack Developer  
**Eduardo Gonzalez** - Data Scientist

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- Dataset: [Letterboxd Dataset](https://www.kaggle.com/datasets/gsimonx37/letterboxd)
- Framework: [Kedro](https://kedro.readthedocs.io)
- Orchestration: [Apache Airflow](https://airflow.apache.org)
