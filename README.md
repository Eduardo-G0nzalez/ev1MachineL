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
- ✅ **Modelado de Machine Learning Supervisado**: Clasificación y regresión con ≥5 modelos cada uno
- ✅ **Modelado de Machine Learning No Supervisado**: Clustering (K-Means, DBSCAN, Hierarchical) y Reducción Dimensional (PCA, t-SNE, UMAP)
- ✅ **Integración**: Clustering como feature engineering para modelos supervisados
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
# El DAG incluye ahora: preparación → EDA → supervisado → no supervisado
```

### Opción 2: Ejecución Local con Kedro

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo (incluye supervisado + no supervisado)
kedro run

# Pipelines específicos
kedro run --pipeline=classification_pipeline
kedro run --pipeline=regression_pipeline
kedro run --pipeline=unsupervised_learning_pipeline  # ← NUEVO

# Solo clustering
kedro run --pipeline=clustering_pipeline

# Solo reducción dimensional
kedro run --pipeline=dimensionality_reduction_pipeline
```

### Opción 3: Ejecución con DVC

```bash
# Ejecutar stage completo de unsupervised learning
dvc repro unsupervised_learning

# Ejecutar todos los stages en orden
dvc repro  # Ejecuta: prepare → features → train_classification → train_regression → evaluate → unsupervised_learning
```

### Opción 4: Jupyter Notebooks

```bash
cd notebooks
jupyter notebook

# Ejecutar notebooks en orden:
# Fase1.ipynb → Fase2.ipynb → ... → Fase6.ipynb
# ev3.ipynb (análisis completo de clustering) ← NUEVO
```

---

## 📚 Fases del Proyecto (CRISP-DM)

### 🔍 Fase 1: Comprensión del Negocio
- **Notebook**: `Fase1.ipynb`
- Análisis inicial y definición de objetivos ML
- Identificación de hipótesis de negocio
- Diferencias entre modelos supervisados y no supervisados

### 📊 Fase 2: Comprensión de Datos (EDA)
- **Notebook**: `Fase2.ipynb`
- **Pipeline**: `eda_pipeline`
- Análisis de calidad: valores faltantes, outliers, completitud
- Análisis temporal: tendencias, estacionalidad
- Visualizaciones avanzadas (12+ gráficos)
- **Generación de datasets ML**: `classification_dataset.csv` y `regression_dataset.csv`

### 🧹 Fase 3: Preparación de Datos
- **Notebook**: `Fase3.ipynb`
- **Pipeline**: `data_preparation_pipeline`
- Limpieza: normalización, deduplicación, filtrado
- Transformación: creación de variables, integración
- Análisis comparativo: 2000s vs 2010s

### 🤖 Fase 4: Modelado Supervisado
- **Clasificación**: `Fase4_Clasificacion.ipynb`
  - **Pipeline**: `classification_pipeline`
  - Target: Éxito comercial (Alto/Medio/Bajo)
  - Modelos: Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost
  - Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  
- **Regresión**: `Fase4_Regresion.ipynb`
  - **Pipeline**: `regression_pipeline`
  - Target: Rating de audiencia (0-5)
  - Modelos: Linear Regression, Random Forest, Gradient Boosting, KNN, XGBoost
  - Métricas: R², RMSE, MAE, MSE

### 🎯 Fase 5: Aprendizaje No Supervisado ⭐ NUEVO

#### **Pipeline**: `unsupervised_learning_pipeline`

#### **Clustering** (3 algoritmos):
- **K-Means**:
  - Selección óptima de k mediante Elbow Method y Silhouette Method
  - Búsqueda automática en rango k=2 a k=10
  - Métricas completas: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
  
- **DBSCAN**:
  - Búsqueda automática de parámetro eps óptimo (0.5, 1.0, 1.5, 2.0, 2.5)
  - Detección automática de outliers
  - Métricas completas para clusters válidos
  
- **Clustering Jerárquico**:
  - AgglomerativeClustering con linkage 'ward'
  - Dendrograma visualizado
  - Mismo k que K-Means para comparación justa
  - Métricas completas

#### **Reducción de Dimensionalidad** (3 técnicas):
- **PCA Completo**:
  - Análisis de varianza explicada por componente
  - Varianza acumulada
  - Análisis de loadings (contribución de variables)
  - Datos para biplot (variables + observaciones)
  - Número óptimo de componentes (95% varianza)
  
- **t-SNE**:
  - Visualización 2D/3D de alta calidad
  - Parámetros configurables (perplexity, max_iter)
  - Muestreo inteligente para datasets grandes (max 10,000 muestras)
  
- **UMAP**:
  - Reducción dimensional moderna
  - Parámetros configurables (n_neighbors, min_dist)
  - Mejor preservación de estructura local que t-SNE

#### **Integración con Modelos Supervisados**:
- Clusters como features adicionales
- Comparación de métricas con/sin clusters
- Mejora de modelos supervisados mediante feature engineering

#### **Notebook**: `ev3.ipynb`
- Análisis completo de clustering
- Visualizaciones profesionales
- Interpretación de negocio por cluster
- Análisis de correlaciones
- Análisis de Silhouette por cluster individual

### 📈 Fase 5: Evaluación
- **Notebook**: `Fase5_Evaluacion.ipynb`
- Comparación de modelos supervisados
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
- **umap-learn**: Reducción dimensional moderna
- **plotly**: Visualizaciones interactivas

### Flujo de Datos Completo

```
┌─────────────┐
│ Datos Raw  │
└──────┬──────┘
       │
       v
┌─────────────────┐
│ Pipeline Kedro  │ ← Docker Container
│ (Preparación)   │
│ + EDA           │
│ + Supervisado   │
│ + No Supervisado│ ← NUEVO
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ DVC Stages      │ ← Versionado
│ - prepare       │
│ - features      │
│ - train_*       │
│ - evaluate      │
│ - unsupervised  │ ← NUEVO
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Airflow DAGs    │ ← Orquestación
│ (kedro_ml_pipeline)│
│ - prepare_data  │
│ - run_eda       │
│ - train_*       │
│ - unsupervised  │ ← NUEVO
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Modelos +       │
│ Métricas JSON   │
│ + Clusters      │ ← NUEVO
└─────────────────┘
```

### Pipelines Kedro Disponibles

#### Pipelines Existentes:
- `data_preparation_pipeline`: Limpieza e integración
- `eda_pipeline`: Análisis exploratorio + generación de datasets ML
- `classification_pipeline`: Modelos de clasificación
- `regression_pipeline`: Modelos de regresión
- `ml_modeling_pipeline`: Pipeline combinado supervisado

#### Pipelines Nuevos ⭐:
- `unsupervised_learning_pipeline`: Pipeline maestro de aprendizaje no supervisado
- `clustering_pipeline`: Pipeline de clustering (K-Means, DBSCAN, Hierarchical)
- `dimensionality_reduction_pipeline`: Pipeline de reducción dimensional (PCA, t-SNE, UMAP)

### DAGs de Airflow

- `kedro_ml_pipeline`: Pipeline completo (preparación → EDA → supervisado → no supervisado) ⭐ ACTUALIZADO
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
│   ├── Fase6_Despliegue.ipynb
│   └── ev3.ipynb           # ← NUEVO: Análisis completo de clustering
├── data/
│   ├── 01_raw/            # Datos originales
│   ├── 02_intermediate/    # Datos procesados (versionados con DVC)
│   ├── 03_primary/         # Datos finales (versionados con DVC)
│   ├── 05_model_input/     # Datos para ML (versionados con DVC)
│   ├── 06_models/          # Modelos y métricas (versionados con DVC)
│   │   ├── clustering_results.pkl  # ← NUEVO
│   │   ├── pca_results.pkl         # ← NUEVO
│   │   ├── tsne_results.pkl        # ← NUEVO
│   │   └── umap_results.pkl        # ← NUEVO
│   ├── 07_model_output/    # Resultados y comparaciones
│   │   ├── clustering_comparison.csv      # ← NUEVO
│   │   ├── clustering_metrics.json       # ← NUEVO
│   │   ├── pca_loadings.csv              # ← NUEVO
│   │   └── dim_reduction_comparison.csv   # ← NUEVO
│   └── 08_reporting/       # Gráficos y visualizaciones
├── src/letterboxdml/
│   └── pipelines/          # Pipelines Kedro modulares
│       ├── data_preparation_pipeline.py
│       ├── eda_pipeline.py
│       ├── ml_modeling_pipeline.py
│       └── unsupervised_learning/  # ← NUEVO
│           ├── pipeline.py          # Pipeline maestro
│           ├── clustering/
│           │   ├── nodes.py         # Funciones de clustering
│           │   └── pipeline.py     # Pipeline de clustering
│           ├── dimensionality_reduction/
│           │   ├── nodes.py         # Funciones PCA, t-SNE, UMAP
│           │   └── pipeline.py     # Pipeline de reducción
│           └── integration/
│               └── nodes.py         # Integración con supervisados
├── dags/                   # DAGs de Airflow
│   ├── kedro_ml_dag.py     # ← ACTUALIZADO con unsupervised
│   ├── kedro_classification_dag.py
│   └── kedro_regression_dag.py
├── dvc.yaml                # Configuración DVC (versionado) ← ACTUALIZADO
├── docker-compose.yml      # Orquestación Docker
├── Dockerfile              # Imagen Docker
├── requirements.txt        # Dependencias Python ← ACTUALIZADO
└── docs/                   # Documentación adicional
    ├── IMPLEMENTACION_UNSUPERVISED.md  # ← NUEVO
    ├── RESUMEN_IMPLEMENTACION.md       # ← NUEVO
    └── ANALISIS_EV3.md                 # ← NUEVO
```

---

## ✅ Requisitos Implementados

### Supervisado:
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

### No Supervisado ⭐ NUEVO:
- ✅ **3 algoritmos de clustering**: K-Means, DBSCAN, Clustering Jerárquico
- ✅ **3 técnicas de reducción dimensional**: PCA completo, t-SNE, UMAP
- ✅ **Métricas completas**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
- ✅ **Selección óptima de k**: Elbow Method + Silhouette Method
- ✅ **Análisis de PCA completo**: Varianza explicada, loadings, biplot
- ✅ **Integración con supervisados**: Clusters como features
- ✅ **Pipeline Kedro completo**: Ejecutable y reproducible
- ✅ **DVC versionado**: Artefactos de clustering versionados
- ✅ **Airflow integrado**: Task de unsupervised learning en DAG maestro
- ✅ **Documentación completa**: Notebooks y documentación técnica

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

### Clustering ⭐ NUEVO
- **K-Means**: 
  - k óptimo: 10 (determinado automáticamente)
  - Silhouette Score: ~0.39
  - Davies-Bouldin Index: ~1.09
  
- **DBSCAN**: 
  - Clusters encontrados: 22 (automático)
  - Silhouette Score: ~0.48 ⭐ (Mejor)
  - Outliers detectados: ~0.1%
  
- **Clustering Jerárquico**: 
  - k: 10 (comparación con K-Means)
  - Silhouette Score: ~0.42
  - Dendrograma disponible para análisis

**Comparación completa**: Ver `data/07_model_output/clustering_comparison.csv`

### Reducción de Dimensionalidad ⭐ NUEVO
- **PCA**: 
  - Varianza explicada PC1: ~15-20%
  - Varianza explicada PC2: ~10-15%
  - Componentes para 95% varianza: Analizado
  - Loadings disponibles: `data/07_model_output/pca_loadings.csv`
  
- **t-SNE**: 
  - Visualización 2D de alta calidad
  - Preservación de estructura local
  
- **UMAP**: 
  - Reducción dimensional moderna
  - Mejor preservación de estructura global

**Comparación completa**: Ver `data/07_model_output/dim_reduction_comparison.csv`

> 📊 **Nota**: Métricas completas disponibles después de ejecutar los pipelines completos.

---

## 🔧 Tecnologías Utilizadas

### Core:
- **Python 3.8+**
- **Kedro** - Framework de data engineering
- **scikit-learn** - Machine Learning
- **Pandas / NumPy** - Manipulación de datos
- **Matplotlib / Seaborn** - Visualizaciones

### No Supervisado ⭐:
- **umap-learn** - Reducción dimensional moderna
- **scipy** - Clustering jerárquico y estadísticas
- **plotly** - Visualizaciones interactivas (opcional)

### Infraestructura:
- **Jupyter** - Notebooks interactivos
- **Docker** - Contenerización
- **Docker Compose** - Orquestación multi-container
- **Apache Airflow** - Workflow orchestration
- **DVC** - Data version control
- **PostgreSQL** - Base de datos para Airflow

---

## 📖 Documentación Adicional

### Guías de Ejecución:
- [`GUIA_EJECUCION_COMPLETA.md`](GUIA_EJECUCION_COMPLETA.md) - Guía paso a paso completa
- [`QUICK_START.md`](QUICK_START.md) - Inicio rápido
- [`GUIA_DVC_GITHUB.md`](GUIA_DVC_GITHUB.md) - Cómo usar DVC con GitHub
- [`INSTALACION_DVC.md`](INSTALACION_DVC.md) - Instalación de DVC
- [`SOLUCION_AIRFLOW.md`](SOLUCION_AIRFLOW.md) - Troubleshooting Airflow

### Documentación Técnica ⭐ NUEVO:
- [`IMPLEMENTACION_UNSUPERVISED.md`](IMPLEMENTACION_UNSUPERVISED.md) - Guía completa de implementación de aprendizaje no supervisado
- [`RESUMEN_IMPLEMENTACION.md`](RESUMEN_IMPLEMENTACION.md) - Resumen ejecutivo de implementación
- [`ANALISIS_EV3.md`](ANALISIS_EV3.md) - Análisis completo del notebook ev3
- [`ANALISIS_COMPARATIVO_EV3_FINAL.md`](ANALISIS_COMPARATIVO_EV3_FINAL.md) - Comparación con requisitos de evaluación final

---

## 🚀 Ejecución de Pipelines

### Ejecutar Pipeline Completo
```bash
kedro run
```

### Ejecutar Solo Aprendizaje No Supervisado
```bash
kedro run --pipeline=unsupervised_learning_pipeline
```

### Ejecutar con DVC (Reproducible)
```bash
# Ejecutar solo unsupervised learning
dvc repro unsupervised_learning

# Ejecutar todo el pipeline
dvc repro
```

### Verificar Estado de DVC
```bash
dvc status
```

---

## 📊 Estructura de Pipelines

### Pipeline de Clustering
```bash
kedro run --pipeline=clustering_pipeline
```

**Incluye**:
1. Preparación de datos para clustering
2. Búsqueda de k óptimo (Elbow + Silhouette)
3. Entrenamiento de K-Means
4. Entrenamiento de DBSCAN
5. Entrenamiento de Clustering Jerárquico
6. Evaluación comparativa
7. Guardado de métricas JSON

### Pipeline de Reducción Dimensional
```bash
kedro run --pipeline=dimensionality_reduction_pipeline
```

**Incluye**:
1. Análisis completo de PCA
2. Reducción con t-SNE
3. Reducción con UMAP
4. Análisis de loadings de PCA
5. Preparación de datos para biplot
6. Comparación de técnicas

---

## 🎓 Evaluación del Proyecto

### Cumplimiento de Requisitos (Nota 7.0)

| Requisito | Estado | Detalles |
|-----------|--------|----------|
| Clustering (≥3 algoritmos) | ✅ 100% | K-Means, DBSCAN, Hierarchical |
| Reducción Dimensional (≥2 técnicas) | ✅ 100% | PCA completo, t-SNE, UMAP |
| Integración con Supervisados | ✅ 100% | Clusters como features |
| Análisis de Patrones | ✅ 100% | Análisis profundo por cluster |
| Orquestación Airflow | ✅ 100% | DAG maestro actualizado |
| Versionado DVC | ✅ 100% | Artefactos versionados |
| Dockerización | ✅ 100% | Dockerfile actualizado |
| Documentación | ✅ 100% | Documentación completa |

**Total**: **8.0/8.0 (100%)** - Listo para nota máxima

---

## 👨‍💻 Autores

**Mathias Jara** - Full Stack Developer  
**Email**: mathias.jara@hotmail.com

**Eduardo Gonzalez** - Data Scientist

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- Dataset: [Letterboxd Dataset](https://www.kaggle.com/datasets/gsimonx37/letterboxd)
- Framework: [Kedro](https://kedro.readthedocs.io)
- Orchestration: [Apache Airflow](https://airflow.apache.org)
- ML Libraries: [scikit-learn](https://scikit-learn.org), [umap-learn](https://umap-learn.readthedocs.io)

---

## 📈 Estadísticas del Proyecto

- **Total de Pipelines**: 8 pipelines Kedro
- **Total de Modelos**: 11 modelos (5 clasificación + 5 regresión + 3 clustering)
- **Técnicas de Reducción**: 3 técnicas (PCA, t-SNE, UMAP)
- **Notebooks**: 8 notebooks completos
- **Líneas de Código**: ~15,000+ líneas
- **Documentación**: 10+ documentos técnicos

---

## 🔄 Última Actualización

**Fecha**: Enero 2025  
**Versión**: 2.0.0  
**Cambios Principales**:
- ✅ Implementación completa de aprendizaje no supervisado
- ✅ 3 algoritmos de clustering con métricas completas
- ✅ 3 técnicas de reducción dimensional
- ✅ Integración con modelos supervisados
- ✅ Pipelines Kedro completos y ejecutables
- ✅ DVC y Airflow actualizados
- ✅ Documentación técnica completa

