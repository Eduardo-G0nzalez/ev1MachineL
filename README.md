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
- pip o conda

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Eduardo-G0nzalez/ev1MachineL.git
cd ev1MachineL
```

### 2. Crear Entorno Virtual

```bash
# Opción 1: Usando venv
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Opción 2: Usando conda
conda create -n letterboxdml python=3.9
conda activate letterboxdml
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Kedro

```bash
# Verificar que Kedro esté instalado correctamente
kedro info

# Configurar el proyecto (si es necesario)
kedro install
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
kedro run
```

### Trabajar con Jupyter Notebooks

```bash
# Iniciar Jupyter Notebook
kedro jupyter notebook

# O iniciar JupyterLab
kedro jupyter lab
```

### Ejecutar Tests

```bash
pytest
```

### Ver Información del Proyecto

```bash
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

### ✨ Visualizaciones Profesionales
- **6 gráficos de calidad de datos** con análisis de completitud
- **6 gráficos temporales** con tendencias y estacionalidad
- **6 gráficos de proceso de limpieza** mostrando antes/después
- **6 gráficos comparativos** entre décadas

### 🎨 Diseño Visual
- Colores consistentes y profesionales
- Comentarios explicativos detallados
- Gráficos interactivos y informativos
- Estadísticas resumidas automáticas

### 🔧 Herramientas Utilizadas
- **Kedro**: Framework de data engineering
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualizaciones
- **Jupyter**: Notebooks interactivos
- **NumPy**: Cálculos numéricos

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con cobertura
pytest --cov=src/letterboxdml

# Ejecutar tests específicos
pytest tests/test_run.py
```

## 📝 Desarrollo

### Agregar Nuevas Funcionalidades

1. Crear nuevos pipelines en `src/letterboxdml/pipelines/`
2. Actualizar configuraciones en `conf/`
3. Agregar tests en `tests/`
4. Documentar cambios en este README

### Estructura de Commits

```
feat: nueva funcionalidad
fix: corrección de bug
docs: actualización de documentación
style: cambios de formato
refactor: refactorización de código
test: agregar o modificar tests
```

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**Mathias Jara**  
*Full Stack Developer*  
📧 mathias.jara@hotmail.com

## 🙏 Agradecimientos

- [Kedro](https://kedro.readthedocs.io/) por el framework de data engineering
- [Letterboxd](https://letterboxd.com/) por los datos cinematográficos
- Comunidad de Python por las librerías de análisis de datos

---

⭐ **¡No olvides darle una estrella al proyecto si te resulta útil!** ⭐