# 🎬 Letterboxd Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Kedro](https://img.shields.io/badge/Kedro-1.0.0-green.svg)](https://kedro.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción del Proyecto

Este proyecto de **Machine Learning** utiliza el framework **Kedro** para analizar datos cinematográficos de **Letterboxd** y estudiar la evolución de géneros cinematográficos entre las décadas de 2000s y 2010s en Estados Unidos.

### 🎯 Objetivos

- **Análisis Exploratorio de Datos (EDA)**: Comprender la estructura y calidad de los datasets cinematográficos
- **Preparación de Datos**: Limpiar y transformar datos para análisis de machine learning
- **Análisis Comparativo**: Comparar tendencias de géneros entre décadas (2000s vs 2010s)
- **Visualizaciones Avanzadas**: Crear gráficos informativos y profesionales

### 📊 Datasets

- **releases**: Eventos de estreno por película y país (1.3M+ registros)
- **genres**: Asignaciones película-género (1M+ registros)
- **countries**: Asociaciones película-país (693K+ registros)

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- Git
- [uv](https://docs.astral.sh/uv/) (Astra) - Gestor de paquetes moderno

### 1. Instalar uv (Astra)

```bash
# En Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# En macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# O usando pip
pip install uv
```

### 2. Clonar el Repositorio

```bash
git clone https://github.com/Eduardo-G0nzalez/ev1MachineL.git
cd ev1MachineL
```

### 3. Crear Entorno Virtual e Instalar Dependencias

```bash
# Crear entorno virtual e instalar todas las dependencias automáticamente
uv sync
```

### 4. Activar Entorno Virtual

```bash
# En Windows
.venv\Scripts\activate

# En macOS/Linux
source .venv/bin/activate
```

### 5. Verificar Instalación

```bash
# Verificar que Kedro esté instalado correctamente
uv run kedro info
```

## 📁 Estructura del Proyecto

```
ev1MachineL/
├── 📁 conf/                    # Configuraciones de Kedro
│   ├── base/                   # Configuración base
│   └── local/                  # Configuración local (no versionado)
├── 📁 data/                    # Datos del proyecto
│   ├── 01_raw/                 # Datos originales
│   ├── 02_intermediate/        # Datos procesados
│   ├── 03_primary/            # Datos principales
│   └── ...
├── 📁 notebooks/              # Jupyter Notebooks
│   ├── Fase1.ipynb           # Análisis inicial
│   ├── Fase2.ipynb           # Comprensión de datos + EDA
│   └── Fase3.ipynb           # Preparación de datos
├── 📁 src/letterboxdml/       # Código fuente del proyecto
├── 📁 tests/                  # Tests unitarios
├── 📄 requirements.txt        # Dependencias Python
├── 📄 pyproject.toml         # Configuración del proyecto
└── 📄 README.md              # Este archivo
```

## 🎮 Uso del Proyecto

### Ejecutar el Pipeline Completo

```bash
# Usando uv (recomendado)
uv run kedro run

# O con entorno virtual activado
kedro run
```

### Trabajar con Jupyter Notebooks

```bash
# Iniciar Jupyter Notebook (método principal)
uv run jupyter notebook --notebook-dir=notebooks --port=8888

# O usando Kedro
uv run kedro jupyter notebook

# Iniciar JupyterLab
uv run kedro jupyter lab
```

### Ejecutar Tests

```bash
# Usando uv
uv run pytest

# O con entorno virtual activado
pytest
```

### Ver Información del Proyecto

```bash
# Usando uv
uv run kedro info

# O con entorno virtual activado
kedro info
```

## 📚 Fases del Análisis

### 🔍 Fase 1: Análisis Inicial
- Carga y exploración básica de datos
- Identificación de problemas de calidad

### 📊 Fase 2: Comprensión de Datos (EDA)
- **Análisis de Calidad de Datos**: Valores faltantes, outliers, completitud
- **Análisis Temporal**: Tendencias, estacionalidad, patrones cíclicos
- **Visualizaciones Avanzadas**: 12+ gráficos informativos con comentarios detallados

### 🧹 Fase 3: Preparación de Datos
- **Limpieza de Datos**: Normalización, deduplicación, filtrado
- **Transformación**: Creación de variables, integración de datasets
- **Análisis Comparativo**: Comparación detallada entre décadas 2000s vs 2010s

## 📈 Características Destacadas

### ✨ Visualizaciones
- **6 gráficos de calidad de datos** con análisis de completitud
- **6 gráficos temporales** con tendencias y estacionalidad
- **6 gráficos de proceso de limpieza** mostrando antes/después
- **6 gráficos comparativos** entre décadas

### 🔧 Herramientas Utilizadas
- **Kedro**: Framework de data engineering
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualizaciones
- **Jupyter**: Notebooks interactivos
- **NumPy**: Cálculos numéricos

<<<<<<< HEAD
## 🧪 Testing

```bash
# Ejecutar todos los tests
uv run pytest

# Ejecutar con cobertura
uv run pytest --cov=src/letterboxdml

# Ejecutar tests específicos
uv run pytest tests/test_run.py

# O con entorno virtual activado
pytest
pytest --cov=src/letterboxdml
pytest tests/test_run.py
```

## ⚡ Inicio Rápido

### Para usuarios que quieren empezar inmediatamente:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Eduardo-G0nzalez/ev1MachineL.git
cd ev1MachineL

# 2. Instalar uv (si no lo tienes)
pip install uv

# 3. Crear entorno virtual e instalar dependencias
uv sync

# 4. Activar entorno virtual
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 5. Iniciar Jupyter Notebooks
uv run jupyter notebook --notebook-dir=notebooks --port=8888
```

**Acceso:** `http://localhost:8888`

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**Mathias Jara**
**Eduardo Gonzalez**


