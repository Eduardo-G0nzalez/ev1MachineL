# Proyecto Machine Learning - Letterboxd

## 📋 Notebooks Disponibles

### Fases del Proyecto (CRISP-DM)
1. **Fase1.ipynb** - Comprensión del negocio y objetivos ML
2. **Fase2.ipynb** - Exploración y análisis exploratorio de datos (EDA)
3. **Fase3.ipynb** - Preparación y limpieza de datos
4. **Fase4_Clasificacion.ipynb** - Modelado de clasificación (5 modelos)
5. **Fase4_Regresion.ipynb** - Modelado de regresión (5 modelos)
6. **Fase5_Evaluacion.ipynb** - Evaluación de modelos y selección
7. **Fase6_Despliegue.ipynb** - Despliegue, monitoreo y conclusiones

---

## 🎯 Hipótesis Implementadas

### Clasificación: Éxito Comercial
- **Target**: Nivel de éxito comercial basado en rating por edad (Alto/Medio/Bajo)
- **Propósito**: Ayudar a distribuidoras a decidir presupuesto de marketing
- **Features**: Duración, década, género (dummy variables)
- **Modelos**: Regresión Logística, Árbol de Decisión, Random Forest, KNN, SVM
- **Métricas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Matrices de Confusión

### Regresión: Rating de Audiencia
- **Target**: Rating numérico de películas (0-5)
- **Propósito**: Plataformas streaming decidir adquisiciones de contenido
- **Features**: Duración, año, género (dummy variables)
- **Modelos**: Linear Regression, Random Forest, Gradient Boosting, KNN, SVR
- **Métricas**: R², RMSE, MAE, MSE, Explained Variance

---

## 🚀 Cómo Ejecutar

> 📖 **Para una guía completa paso a paso, ver**: [`GUIA_EJECUCION_COMPLETA.md`](GUIA_EJECUCION_COMPLETA.md)

### Opción 1: Jupyter Notebook
```bash
cd ev1MachineL/notebooks
jupyter notebook
```

### Opción 2: Kedro Pipelines
```bash
# Pipeline completo
kedro run

# Pipeline específico
kedro run --pipeline=classification_pipeline
kedro run --pipeline=regression_pipeline
kedro run --pipeline=ml_modeling_pipeline
```

### Opción 3: Con Docker
```bash
# Construir imagen
docker build -t kedro-ml .

# Ejecutar pipeline completo
docker run --rm -v $(pwd)/data:/app/data kedro-ml kedro run

# Ejecutar con docker-compose (incluye Airflow)
docker-compose up -d
```

### Opción 4: Con DVC (Versionado de Datos)
```bash
# Reproducir pipeline completo
dvc repro

# Reproducir stage específico
dvc repro train_classification
dvc repro train_regression

# Ver métricas
dvc metrics show
```

### Opción 5: Con Airflow (Orquestación)
```bash
# Iniciar servicios (Airflow + Postgres)
docker-compose up -d

# Acceder a Airflow UI
# http://localhost:8080
# Usuario: admin
# Password: admin

# Activar DAG manualmente desde la UI
```

---

## 📊 Estructura del Proyecto

```
ev1MachineL/
├── notebooks/          # Notebooks Fase 1-6 (CRISP-DM completo)
├── data/
│   ├── 01_raw/        # Datos originales
│   ├── 02_intermediate/# Datos procesados
│   ├── 03_primary/     # Datos finales
│   ├── 05_model_input/ # Datos para ML
│   ├── 06_models/      # Modelos y métricas
│   └── 08_reporting/   # Gráficos y visualizaciones
├── src/
│   └── letterboxdml/
│       └── pipelines/   # Pipelines Kedro
└── requirements.txt     # Dependencias
```

---

## ✅ Datos Disponibles

- **movies.csv**: 941,597 películas
- **releases.csv**: 1,332,782 estrenos
- **countries.csv**: 693,476 países
- **genres.csv**: 1,046,849 géneros

---

---

## 🔧 Arquitectura del Sistema

### Componentes Principales

1. **Kedro**: Orquestación de pipelines
2. **DVC**: Versionado de datos y modelos
3. **Airflow**: Orquestación automatizada
4. **Docker**: Contenerización y reproducibilidad
5. **scikit-learn**: Machine Learning con GridSearchCV

### Flujo de Datos

```
┌─────────────┐
│ Datos Raw   │
└──────┬──────┘
       │
       v
┌─────────────────┐
│ Pipeline Kedro  │
│ (Preparación)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ DVC Stages      │
│ (Versionado)    │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Airflow DAGs    │
│ (Orquestación)  │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Modelos        │
│ + Métricas      │
└─────────────────┘
```

### Estructura de Pipelines

- **data_preparation_pipeline**: Limpieza e integración de datos
- **eda_pipeline**: Análisis exploratorio
- **classification_pipeline**: Entrenar modelos de clasificación
- **regression_pipeline**: Entrenar modelos de regresión
- **ml_modeling_pipeline**: Pipeline combinado

---

## 📊 Requisitos Implementados ✅

- ✅ **Metodología CRISP-DM** completa (Fases 1-6)
- ✅ **Pipelines Kedro** modulares y ejecutables
- ✅ **DVC** para versionado de datos, features y modelos
- ✅ **Airflow** con DAGs orquestados
- ✅ **Docker** con imagen reproducible
- ✅ **≥5 modelos** por tipo (clasificación y regresión)
- ✅ **GridSearchCV** con validación cruzada (k≥5)
- ✅ **Métricas completas** con mean±std
- ✅ **Tabla comparativa** de resultados
- ✅ **Evaluación y selección** de mejores modelos (Fase 5)
- ✅ **Plan de despliegue** y monitoreo (Fase 6)
- ✅ **Documentación técnica** completa

---

## 🎯 Resultados Finales

### Clasificación (Éxito Comercial)
- **Mejor modelo**: XGBoost
- **Accuracy**: 75.7%
- **F1-Score**: 0.7838
- **Estado**: ✅ Cumple objetivos

### Regresión (Rating de Audiencia)
- **Mejor modelo**: Random Forest
- **R² Score**: 43.6%
- **RMSE**: 0.3343
- **Estado**: ⚠️ Aceptable (limitado por baja varianza del target)

---

## 🎓 Autores
**Mathias Jara** - Full Stack Developer

**Eduardo Gonzalez** - Data Scientist

