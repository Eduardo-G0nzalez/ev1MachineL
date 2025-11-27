"""
Pipeline para Fase 2: Análisis Exploratorio de Datos (EDA).

Este módulo contiene todas las funciones necesarias para realizar un análisis exploratorio
completo de los datos cinematográficos, incluyendo:

1. Análisis básico de calidad de datos
2. Medidas estadísticas descriptivas (tendencia central, dispersión, posición)
3. Visualizaciones de calidad y patrones temporales
4. Análisis avanzado de variables categóricas (géneros)
5. Métricas de diversidad y concentración
6. Análisis de asociaciones entre géneros
7. Tendencias temporales y estacionalidad

Autor: Mathias Jara y Eduardo Gonzalez
Fecha: 2025
"""

# =============================================================================
# IMPORTS Y CONFIGURACIÓN
# =============================================================================

from kedro.pipeline import Pipeline, node, pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# Imports para análisis estadísticos avanzados
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Importar funciones de preparación de datos para ML
from .ml_modeling_pipeline import prepare_classification_data, prepare_regression_data


# =============================================================================
# FUNCIONES DE ANÁLISIS BÁSICO
# =============================================================================

def load_and_display_datasets(releases, genres, countries):
    """
    Carga y muestra información básica de los datasets.
    
    Esta función realiza un análisis inicial de los tres datasets principales:
    - releases: Eventos de estreno por película y país
    - genres: Asignaciones película-género  
    - countries: Asociaciones película-país
    
    Args:
        releases (pd.DataFrame): Dataset de estrenos
        genres (pd.DataFrame): Dataset de géneros
        countries (pd.DataFrame): Dataset de países
        
    Returns:
        dict: Diccionario con información básica de cada dataset
    """
    # Organizar datasets en diccionario para procesamiento iterativo
    datasets = {
        "releases": releases,
        "genres": genres, 
        "countries": countries
    }
    
    results = {}
    
    # Procesar cada dataset individualmente
    for name, df in datasets.items():
        print(f"\n=== DATASET: {name} ===")
        print("shape:", df.shape)
        
        # Mostrar primeras 5 filas para inspección visual
        display(df.head(5))
        
        # Crear esquema de tipos de datos
        schema = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str)
        })
        print("dtypes por columna:")
        display(schema)
        
        # Almacenar información para uso posterior
        results[f"{name}_info"] = {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "sample": df.head(5)
        }
    
    return results


def calculate_central_tendency(releases, genres, countries):
    """
    Calcula medidas de tendencia central para cada dataset.
    
    Esta función analiza las medidas de tendencia central (media, mediana, moda)
    para variables numéricas, categóricas y de fecha en cada dataset.
    
    Args:
        releases (pd.DataFrame): Dataset de estrenos
        genres (pd.DataFrame): Dataset de géneros
        countries (pd.DataFrame): Dataset de países
        
    Returns:
        dict: Confirmación de análisis completado
    """
    
    def central_tendency_by_dataset(df, dataset_name, date_col=None):
        """
        Función auxiliar para calcular tendencia central por dataset.
        
        Args:
            df (pd.DataFrame): Dataset a analizar
            dataset_name (str): Nombre del dataset
            date_col (str, optional): Columna de fecha a analizar
        """
        print("="*40)
        print(f"\n=== Medidas de tendencia central — {dataset_name} ===")
        print("="*40)
        
        # === ANÁLISIS DE VARIABLES NUMÉRICAS ===
        # Seleccionar columnas numéricas excluyendo 'id' (no relevante para análisis)
        num = df.select_dtypes(include=[np.number]).drop(columns=["id"], errors="ignore")
        
        if num.shape[1] > 0:
            # Calcular media y mediana para variables numéricas
            out_num = pd.DataFrame({
                "mean": num.mean(numeric_only=True),
                "median": num.median(numeric_only=True)
            })
            print("Numéricas (media, mediana):")
            display(out_num.round(4))
        else:
            print("Numéricas: (sin columnas numéricas relevantes)")
        
        # === ANÁLISIS DE VARIABLES CATEGÓRICAS ===
        # Seleccionar columnas de tipo objeto (categóricas)
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        
        if cat_cols:
            rows, n = [], len(df)
            for c in cat_cols:
                # Calcular moda (valor más frecuente) y sus estadísticas
                vc = df[c].value_counts(dropna=False)
                mode_val = vc.index[0] if len(vc) else None
                mode_freq = int(vc.iloc[0]) if len(vc) else 0
                mode_share = (mode_freq / n) if n else np.nan
                
                rows.append({
                    "column": c, 
                    "mode": mode_val, 
                    "mode_freq": mode_freq, 
                    "mode_share": round(mode_share, 4)
                })
            
            print("Categóricas (moda, frecuencia, participación):")
            display(pd.DataFrame(rows).sort_values("mode_share", ascending=False))
        else:
            print("Categóricas: (sin columnas categóricas)")
        
        # === ANÁLISIS DE VARIABLES DE FECHA ===
        if date_col and date_col in df.columns:
            # Convertir a datetime y calcular mediana temporal
            s = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
            med = s.median() if s.notna().any() else None
            print(f"Mediana de fecha ({date_col}): {med}")
    
    # Aplicar análisis a cada dataset
    central_tendency_by_dataset(releases, "releases", date_col="date")
    central_tendency_by_dataset(genres, "genres")
    central_tendency_by_dataset(countries, "countries")
    
    return {"central_tendency_analysis": "completed"}


def calculate_dispersion(releases, genres, countries):
    """
    Calcula medidas de dispersión para cada dataset.
    
    Esta función analiza la variabilidad y dispersión de los datos mediante
    métricas como varianza, desviación estándar, rango, coeficiente de variación
    e intervalo intercuartílico (IQR).
    
    Args:
        releases (pd.DataFrame): Dataset de estrenos
        genres (pd.DataFrame): Dataset de géneros
        countries (pd.DataFrame): Dataset de países
        
    Returns:
        dict: Confirmación de análisis completado
    """
    
    def _numeric(df):
        """
        Selecciona columnas numéricas relevantes (excluye 'id').
        
        Args:
            df (pd.DataFrame): Dataset a procesar
            
        Returns:
            pd.DataFrame: Columnas numéricas sin 'id'
        """
        return df.select_dtypes(include=[np.number]).drop(columns=["id"], errors="ignore")

    def _metrics(series: pd.Series) -> pd.Series:
        """
        Calcula métricas de dispersión para una serie numérica.
        
        Métricas calculadas:
        - var: Varianza (ddof=1 para muestra)
        - std: Desviación estándar (ddof=1 para muestra)
        - range: Rango (max - min)
        - cv: Coeficiente de variación (std/mean)
        - iqr: Intervalo intercuartílico (Q75 - Q25)
        
        Args:
            series (pd.Series): Serie numérica a analizar
            
        Returns:
            pd.Series: Métricas de dispersión
        """
        s = series.dropna()
        if s.empty:
            return pd.Series({"var": np.nan, "std": np.nan, "range": np.nan, "cv": np.nan, "iqr": np.nan})
        
        # Calcular métricas básicas
        var = s.var()  # ddof=1 (muestra)
        std = s.std()  # ddof=1
        rng = s.max() - s.min()
        mean = s.mean()
        
        # Coeficiente de variación (manejo de división por cero)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = std / mean if mean not in (0, np.nan) else np.nan
        
        # Intervalo intercuartílico
        q75, q25 = s.quantile(0.75), s.quantile(0.25)
        iqr = q75 - q25
        
        return pd.Series({"var": var, "std": std, "range": rng, "cv": cv, "iqr": iqr})

    def dispersion_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea tabla de dispersión para todas las columnas numéricas.
        
        Args:
            df (pd.DataFrame): Dataset a analizar
            
        Returns:
            pd.DataFrame: Tabla con métricas de dispersión por columna
        """
        num = _numeric(df)
        if num.shape[1] == 0:
            return pd.DataFrame({"note": ["(sin columnas numéricas relevantes)"]})
        
        rows = []
        for col in num.columns:
            m = _metrics(num[col])
            m.name = col
            rows.append(m)
        return pd.DataFrame(rows).round(6)

    def _date_to_days(df: pd.DataFrame, date_col: str):
        """
        Convierte columna de fecha a días para calcular dispersión temporal.
        
        Args:
            df (pd.DataFrame): Dataset con columna de fecha
            date_col (str): Nombre de la columna de fecha
            
        Returns:
            pd.Series: Fechas convertidas a días (float)
        """
        if date_col not in df.columns:
            return None
        
        # Convertir a datetime y eliminar valores nulos
        s = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None).dropna()
        if s.empty:
            return None
        
        # Convertir nanosegundos a días (ns -> s -> días)
        days = (s.astype("int64") / 1e9) / 86400.0
        return pd.Series(days, index=s.index)

    # === ANÁLISIS DE DISPERSIÓN POR DATASET ===
    
    print("="*40)
    print("=== DISPERSIÓN — releases (numéricas) ===")
    print("="*40)
    display(dispersion_table(releases))

    # Análisis especial: dispersión temporal de fechas
    print("="*40)
    print("\n=== DISPERSIÓN — releases.fecha (en días) ===")
    print("="*40)
    rel_days = _date_to_days(releases, "date")
    if rel_days is not None:
        display(pd.DataFrame([_metrics(rel_days)], index=["date_days"]).round(6))
    else:
        print("(columna 'date' no parseable o ausente)")

    print("="*40)
    print("\n=== DISPERSIÓN — genres (numéricas) ===")
    print("="*40)
    display(dispersion_table(genres))

    print("="*40)
    print("\n=== DISPERSIÓN — countries (numéricas) ===")
    print("="*40)
    display(dispersion_table(countries))

    return {"dispersion_analysis": "completed"}


def calculate_position_measures(releases, genres, countries):
    """
    Calcula medidas de posición para cada dataset.
    
    Esta función analiza las medidas de posición (percentiles, cuartiles, min/max)
    para variables numéricas y de fecha en cada dataset.
    
    Args:
        releases (pd.DataFrame): Dataset de estrenos
        genres (pd.DataFrame): Dataset de géneros
        countries (pd.DataFrame): Dataset de países
        
    Returns:
        dict: Confirmación de análisis completado
    """
    
    def _numeric(df):
        """
        Selecciona columnas numéricas relevantes (excluye 'id').
        
        Args:
            df (pd.DataFrame): Dataset a procesar
            
        Returns:
            pd.DataFrame: Columnas numéricas sin 'id'
        """
        return df.select_dtypes(include=[np.number]).drop(columns=["id"], errors="ignore")

    def numeric_position_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea tabla de medidas de posición para variables numéricas.
        
        Incluye: count, sum, min, p10, p25, median, p75, p90, max
        
        Args:
            df (pd.DataFrame): Dataset a analizar
            
        Returns:
            pd.DataFrame: Tabla con medidas de posición por columna
        """
        num = _numeric(df)
        if num.shape[1] == 0:
            return pd.DataFrame({"note": ["(sin columnas numéricas relevantes)"]})
        
        # Calcular estadísticas descriptivas con percentiles específicos
        desc = num.describe(percentiles=[.10, .25, .50, .75, .90]).T
        
        # Renombrar columnas para mayor claridad
        desc = desc.rename(columns={
            "count": "count", "mean": "mean", "std": "std", "min": "min",
            "10%": "p10", "25%": "p25", "50%": "median", 
            "75%": "p75", "90%": "p90", "max": "max"
        })
        
        # Agregar suma total
        desc["sum"] = num.sum()
        
        # Ordenar columnas en orden lógico de posición
        cols = ["count", "sum", "min", "p10", "p25", "median", "p75", "p90", "max"]
        return desc[cols].round(6)

    def date_position_table(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Crea tabla de medidas de posición para variables de fecha.
        
        Args:
            df (pd.DataFrame): Dataset con columna de fecha
            date_col (str): Nombre de la columna de fecha
            
        Returns:
            pd.DataFrame: Tabla con medidas de posición temporal
        """
        if date_col not in df.columns:
            return pd.DataFrame({"note": [f"(columna '{date_col}' no existe)"]})
        
        # Convertir a datetime y eliminar valores nulos
        s = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None).dropna()
        if s.empty:
            return pd.DataFrame({"note": [f"(no parseable '{date_col}')"]})
        
        # Calcular percentiles temporales
        qs = s.quantile([.10, .25, .50, .75, .90])
        
        out = pd.DataFrame({
            "count_valid": [s.shape[0]],
            "min": [s.min()],
            "p10": [qs.loc[0.10]],
            "p25": [qs.loc[0.25]],
            "median": [qs.loc[0.50]],
            "p75": [qs.loc[0.75]],
            "p90": [qs.loc[0.90]],
            "max": [s.max()]
        }, index=[date_col])
        return out

    # === ANÁLISIS DE MEDIDAS DE POSICIÓN POR DATASET ===
    
    print("========================================")
    print("=== POSICIÓN — releases (numéricas) ===")
    print("========================================")
    display(numeric_position_table(releases))

    print("\n=== POSICIÓN — releases.fecha (date) ===")
    display(date_position_table(releases, "date"))

    print("\n=======================================")
    print("=== POSICIÓN — genres (numéricas) ===")
    print("=======================================")
    display(numeric_position_table(genres))

    print("\n=========================================")
    print("=== POSICIÓN — countries (numéricas) ===")
    print("=========================================")
    display(numeric_position_table(countries))

    return {"position_analysis": "completed"}


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def create_quality_visualizations(releases, genres, countries):
    """
    Crea visualizaciones de calidad de datos.
    
    Esta función genera 6 gráficos que analizan la calidad de los datos:
    1. Valores faltantes por columna y dataset
    2. Mapa de calor de completitud
    3. Distribución de tipos de lanzamiento
    4. Top 10 países por estrenos
    5. Análisis de outliers temporales
    6. Top 15 géneros más frecuentes
    
    Args:
        releases (pd.DataFrame): Dataset de estrenos
        genres (pd.DataFrame): Dataset de géneros
        countries (pd.DataFrame): Dataset de países
        
    Returns:
        dict: Confirmación de visualizaciones completadas
    """
    
    # === CONFIGURACIÓN DE GRÁFICOS ===
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10

    # Crear figura con subplots para análisis de calidad
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Análisis de Calidad de Datos - Valores Faltantes y Completitud', 
                 fontsize=16, fontweight='bold')

    # === 1. ANÁLISIS DE VALORES FALTANTES ===
    datasets = ['releases', 'genres', 'countries']
    missing_data = []

    # Calcular valores faltantes para cada dataset
    for name in datasets:
        df = releases if name == 'releases' else genres if name == 'genres' else countries
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        for col in df.columns:
            missing_data.append({
                'dataset': name,
                'column': col,
                'missing_count': missing_count[col],
                'missing_pct': missing_pct[col]
            })

    missing_df = pd.DataFrame(missing_data)

    # Gráfico de barras - Valores faltantes por columna
    missing_pivot = missing_df.pivot(index='column', columns='dataset', values='missing_pct')
    missing_pivot.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0,0].set_title('Porcentaje de Valores Faltantes por Columna')
    axes[0,0].set_ylabel('Porcentaje (%)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend(title='Dataset')
    axes[0,0].grid(True, alpha=0.3)

    # === 2. MAPA DE CALOR DE COMPLETITUD ===
    completeness_pivot = 100 - missing_pivot
    sns.heatmap(completeness_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=axes[0,1], cbar_kws={'label': 'Completitud (%)'})
    axes[0,1].set_title('Mapa de Calor - Completitud por Dataset y Columna')
    axes[0,1].set_xlabel('Dataset')
    axes[0,1].set_ylabel('Columna')

    # === 3. DISTRIBUCIÓN DE TIPOS DE LANZAMIENTO ===
    type_counts = releases['type'].value_counts()
    axes[0,2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0,2].set_title('Distribución de Tipos de Lanzamiento')

    # === 4. TOP 10 PAÍSES POR ESTRENOS ===
    country_counts = releases['country'].value_counts().head(10)
    axes[1,0].barh(range(len(country_counts)), country_counts.values, 
                   color='lightblue', alpha=0.7)
    axes[1,0].set_yticks(range(len(country_counts)))
    axes[1,0].set_yticklabels(country_counts.index)
    axes[1,0].set_title('Top 10 Países por Número de Estrenos')
    axes[1,0].set_xlabel('Número de Estrenos')
    axes[1,0].grid(True, alpha=0.3)

    # === 5. ANÁLISIS DE OUTLIERS TEMPORALES ===
    releases_temp = releases.copy()
    releases_temp['date_parsed'] = pd.to_datetime(releases_temp['date'], errors='coerce')
    releases_temp = releases_temp.dropna(subset=['date_parsed'])
    releases_temp['year'] = releases_temp['date_parsed'].dt.year

    # Filtrar años razonables para el análisis de outliers
    yearly_counts = releases_temp.groupby('year').size()
    axes[1,1].plot(yearly_counts.index, yearly_counts.values, 
                   marker='o', linewidth=1, markersize=3)
    axes[1,1].set_title('Distribución de Estrenos por Año (Análisis de Outliers)')
    axes[1,1].set_xlabel('Año')
    axes[1,1].set_ylabel('Número de Estrenos')
    axes[1,1].grid(True, alpha=0.3)

    # === 6. DISTRIBUCIÓN DE GÉNEROS (TOP 15) ===
    genre_counts = genres['genre'].value_counts().head(15)
    axes[1,2].bar(range(len(genre_counts)), genre_counts.values, 
                  color='lightgreen', alpha=0.7)
    axes[1,2].set_xticks(range(len(genre_counts)))
    axes[1,2].set_xticklabels(genre_counts.index, rotation=45, ha='right')
    axes[1,2].set_title('Top 15 Géneros Más Frecuentes')
    axes[1,2].set_ylabel('Número de Películas')
    axes[1,2].grid(True, alpha=0.3)

    # === GUARDAR Y MOSTRAR GRÁFICOS ===
    plt.tight_layout()
    plt.savefig('data/08_reporting/quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # === RESUMEN ESTADÍSTICO DE CALIDAD ===
    print("=== RESUMEN DE CALIDAD DE DATOS ===")
    print(f"Total de registros en releases: {len(releases):,}")
    print(f"Total de registros en genres: {len(genres):,}")
    print(f"Total de registros en countries: {len(countries):,}")
    print(f"\nValores faltantes en rating: {releases['rating'].isnull().sum():,} "
          f"({releases['rating'].isnull().mean()*100:.1f}%)")
    print(f"Rango temporal: {releases_temp['year'].min()}-{releases_temp['year'].max()}")

    return {"quality_visualizations": "completed"}


def create_temporal_analysis(releases):
    """
    Crea análisis temporal y visualizaciones.
    
    Esta función genera 6 gráficos que analizan patrones temporales:
    1. Tendencia anual de estrenos (2000-2020)
    2. Distribución de estrenos por mes
    3. Distribución de estrenos por trimestre
    4. Distribución de estrenos por día de la semana
    5. Comparación de estrenos por década (2000s vs 2010s)
    6. Top 5 países por década
    
    Args:
        releases (pd.DataFrame): Dataset de estrenos
        
    Returns:
        dict: Confirmación de análisis temporal completado
    """
    
    # === CONFIGURACIÓN DE GRÁFICOS ===
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (18, 12)
    plt.rcParams['font.size'] = 10

    # === PREPARACIÓN DE DATOS TEMPORALES ===
    releases_temp = releases.copy()
    releases_temp['date_parsed'] = pd.to_datetime(releases_temp['date'], errors='coerce')
    releases_temp = releases_temp.dropna(subset=['date_parsed'])
    
    # Extraer componentes temporales
    releases_temp['year'] = releases_temp['date_parsed'].dt.year
    releases_temp['month'] = releases_temp['date_parsed'].dt.month
    releases_temp['quarter'] = releases_temp['date_parsed'].dt.quarter
    releases_temp['day_of_week'] = releases_temp['date_parsed'].dt.day_name()

    # Filtrar años razonables (1900-2025) para eliminar outliers temporales
    releases_temp = releases_temp[(releases_temp['year'] >= 1900) & (releases_temp['year'] <= 2025)]

    # Crear figura con subplots para análisis temporal
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('Análisis de Tendencias Temporales y Patrones Estacionales', 
                 fontsize=16, fontweight='bold')

    # === 1. TENDENCIA ANUAL (2000-2020) ===
    releases_2000_2020 = releases_temp[(releases_temp['year'] >= 2000) & (releases_temp['year'] <= 2020)]
    yearly_counts = releases_2000_2020.groupby('year').size()

    axes[0,0].plot(yearly_counts.index, yearly_counts.values, 
                   marker='o', linewidth=2, markersize=6, color='blue')
    axes[0,0].set_title('Tendencia de Estrenos Anuales (2000-2020)')
    axes[0,0].set_xlabel('Año')
    axes[0,0].set_ylabel('Número de Estrenos')
    axes[0,0].grid(True, alpha=0.3)

    # === 2. DISTRIBUCIÓN POR MES ===
    monthly_counts = releases_temp.groupby('month').size()
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    bars = axes[0,1].bar(range(1, 13), monthly_counts.values, color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Distribución de Estrenos por Mes')
    axes[0,1].set_xlabel('Mes')
    axes[0,1].set_ylabel('Número de Estrenos')
    axes[0,1].set_xticks(range(1, 13))
    axes[0,1].set_xticklabels(month_names)
    axes[0,1].grid(True, alpha=0.3)

    # === 3. DISTRIBUCIÓN POR TRIMESTRE ===
    quarterly_counts = releases_temp.groupby('quarter').size()
    quarter_labels = ['Q1 (Ene-Mar)', 'Q2 (Abr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dic)']
    axes[1,0].pie(quarterly_counts.values, labels=quarter_labels, autopct='%1.1f%%', startangle=90, 
                  colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    axes[1,0].set_title('Distribución de Estrenos por Trimestre')

    # === 4. DISTRIBUCIÓN POR DÍA DE LA SEMANA ===
    dow_counts = releases_temp.groupby('day_of_week').size()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = dow_counts.reindex(dow_order)
    dow_spanish = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    bars = axes[1,1].bar(range(len(dow_counts)), dow_counts.values, 
                         color='lightgreen', alpha=0.7)
    axes[1,1].set_title('Distribución de Estrenos por Día de la Semana')
    axes[1,1].set_xlabel('Día de la Semana')
    axes[1,1].set_ylabel('Número de Estrenos')
    axes[1,1].set_xticks(range(len(dow_counts)))
    axes[1,1].set_xticklabels(dow_spanish, rotation=45)
    axes[1,1].grid(True, alpha=0.3)

    # === 5. ANÁLISIS DE DÉCADAS (2000s vs 2010s) ===
    decade_2000s = releases_temp[(releases_temp['year'] >= 2000) & (releases_temp['year'] <= 2009)]
    decade_2010s = releases_temp[(releases_temp['year'] >= 2010) & (releases_temp['year'] <= 2019)]

    decade_data = [len(decade_2000s), len(decade_2010s)]
    decade_labels = ['2000s', '2010s']
    colors = ['lightblue', 'lightcoral']

    bars = axes[2,0].bar(decade_labels, decade_data, color=colors, alpha=0.7)
    axes[2,0].set_title('Comparación de Estrenos por Década')
    axes[2,0].set_ylabel('Número de Estrenos')
    axes[2,0].grid(True, alpha=0.3)

    # === 6. TOP 5 PAÍSES POR DÉCADA ===
    top_countries_2000s = releases_temp[(releases_temp['year'] >= 2000) & (releases_temp['year'] <= 2009)]['country'].value_counts().head(5)
    top_countries_2010s = releases_temp[(releases_temp['year'] >= 2010) & (releases_temp['year'] <= 2019)]['country'].value_counts().head(5)

    x = np.arange(len(top_countries_2000s))
    width = 0.35

    axes[2,1].bar(x - width/2, top_countries_2000s.values, width, 
                  label='2000s', alpha=0.7, color='lightblue')
    axes[2,1].bar(x + width/2, top_countries_2010s.values, width, 
                  label='2010s', alpha=0.7, color='lightcoral')
    axes[2,1].set_title('Top 5 Países por Número de Estrenos (Comparación por Década)')
    axes[2,1].set_xlabel('País')
    axes[2,1].set_ylabel('Número de Estrenos')
    axes[2,1].set_xticks(x)
    axes[2,1].set_xticklabels(top_countries_2000s.index, rotation=45, ha='right')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)

    # === GUARDAR Y MOSTRAR GRÁFICOS ===
    plt.tight_layout()
    plt.savefig('data/08_reporting/temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {"temporal_analysis": "completed"}


def create_genre_analysis(genres):
    """
    Crea análisis de géneros y visualizaciones.
    
    Esta función genera 4 gráficos que analizan la distribución de géneros:
    1. Top 15 géneros más frecuentes (barras horizontales)
    2. Distribución de géneros (gráfico de pastel)
    3. Boxplot de géneros por película
    4. Histograma de distribución de géneros por película
    
    Args:
        genres (pd.DataFrame): Dataset de géneros
        
    Returns:
        dict: Confirmación de análisis de géneros completado
    """
    
    # === CONFIGURACIÓN DE GRÁFICOS ===
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10

    # === PREPARACIÓN DE DATOS ===
    # Top 15 géneros más frecuentes
    top_genres = genres['genre'].value_counts().head(15)

    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Géneros Cinematográficos', fontsize=16, fontweight='bold')

    # === 1. GRÁFICO DE BARRAS HORIZONTALES - TOP 15 GÉNEROS ===
    axes[0,0].barh(range(len(top_genres)), top_genres.values, 
                   color='lightgreen', alpha=0.7)
    axes[0,0].set_yticks(range(len(top_genres)))
    axes[0,0].set_yticklabels(top_genres.index)
    axes[0,0].set_title('Top 15 Géneros Más Frecuentes')
    axes[0,0].set_xlabel('Número de Películas')
    axes[0,0].grid(True, alpha=0.3)

    # === 2. GRÁFICO DE PASTEL - TOP 10 GÉNEROS ===
    top_10_genres = genres['genre'].value_counts().head(10)
    other_count = genres['genre'].value_counts().iloc[10:].sum()
    top_10_with_other = top_10_genres.copy()
    top_10_with_other['Otros'] = other_count

    axes[0,1].pie(top_10_with_other.values, labels=top_10_with_other.index, 
                  autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('Distribución de Géneros (Top 10 + Otros)')

    # === 3. BOXPLOT DE DISTRIBUCIÓN DE GÉNEROS POR PELÍCULA ===
    movies_genre_count = genres.groupby('id').size()
    axes[1,0].boxplot([movies_genre_count.values], labels=['Géneros por Película'])
    axes[1,0].set_title('Distribución de Número de Géneros por Película')
    axes[1,0].set_ylabel('Número de Géneros')
    axes[1,0].grid(True, alpha=0.3)

    # === 4. HISTOGRAMA DE GÉNEROS POR PELÍCULA ===
    axes[1,1].hist(movies_genre_count.values, bins=20, alpha=0.7, 
                   color='orange', edgecolor='black')
    axes[1,1].set_title('Distribución de Géneros por Película')
    axes[1,1].set_xlabel('Número de Géneros')
    axes[1,1].set_ylabel('Número de Películas')
    axes[1,1].grid(True, alpha=0.3)

    # === GUARDAR Y MOSTRAR GRÁFICOS ===
    plt.tight_layout()
    plt.savefig('data/08_reporting/genre_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # === RESUMEN ESTADÍSTICO ===
    print(f"Total de géneros únicos: {genres['genre'].nunique()}")
    print(f"Total de películas: {genres['id'].nunique():,}")
    print(f"Promedio de géneros por película: {movies_genre_count.mean():.2f}")
    print(f"Mediana de géneros por película: {movies_genre_count.median():.0f}")
    print(f"Género más común: {top_genres.index[0]} ({top_genres.iloc[0]:,} películas)")

    return {"genre_analysis": "completed"}


# =============================================================================
# FUNCIONES DE ANÁLISIS CATEGÓRICO AVANZADO
# =============================================================================

def analyze_genre_distribution_by_decade(final_df):
    """
    Análisis detallado de distribución de géneros por década.
    
    Esta función realiza un análisis estadístico completo de la distribución
    de géneros entre las décadas 2000s y 2010s, incluyendo:
    - Tabla de contingencia
    - Proporciones relativas
    - Cambios porcentuales
    - Test de Chi-cuadrado para independencia
    
    Args:
        final_df (pd.DataFrame): Dataset final con datos integrados
        
    Returns:
        dict: Resultados del análisis de distribución por década
    """
    print("=== ANÁLISIS DE DISTRIBUCIÓN DE GÉNEROS POR DÉCADA ===")
    
    # === 1. TABLA DE CONTINGENCIA GÉNEROS VS DÉCADAS ===
    contingency_table = pd.crosstab(final_df['genre'], final_df['decade'], margins=True)
    print("\nTabla de Contingencia (Géneros vs Décadas):")
    display(contingency_table)
    
    # === 2. PROPORCIONES RELATIVAS POR DÉCADA ===
    proportions = pd.crosstab(final_df['genre'], final_df['decade'], normalize='index')
    print("\nProporciones Relativas por Década:")
    display(proportions.round(4))
    
    # === 3. CAMBIOS PORCENTUALES ENTRE DÉCADAS ===
    changes = ((proportions['2010s'] - proportions['2000s']) / proportions['2000s'] * 100)
    changes = changes.dropna().sort_values(ascending=False)
    print("\nCambios Porcentuales (2000s → 2010s):")
    display(changes.round(2))
    
    # === 4. TEST DE CHI-CUADRADO PARA INDEPENDENCIA ===
    chi2, p_value, dof, expected = chi2_contingency(contingency_table.iloc[:-1, :-1])
    print(f"\nTest de Chi-cuadrado:")
    print(f"Chi2: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Grados de libertad: {dof}")
    print(f"Significativo (p<0.05): {'Sí' if p_value < 0.05 else 'No'}")
    
    return {
        'contingency_table': contingency_table,
        'proportions': proportions,
        'changes': changes,
        'chi2_test': {'chi2': chi2, 'p_value': p_value, 'dof': dof}
    }


def analyze_genre_diversity(final_df):
    """
    Análisis de diversidad y concentración de géneros.
    
    Esta función calcula métricas de diversidad para comparar la concentración
    de géneros entre las décadas 2000s y 2010s, incluyendo:
    - Índice de diversidad de Shannon
    - Índice de diversidad de Simpson
    - Índice de Gini (concentración)
    - Número de géneros únicos
    - Total de películas
    
    Args:
        final_df (pd.DataFrame): Dataset final con datos integrados
        
    Returns:
        dict: Métricas de diversidad por década
    """
    print("=== ANÁLISIS DE DIVERSIDAD DE GÉNEROS ===")
    
    def shannon_diversity(series):
        """
        Calcula el índice de diversidad de Shannon.
        
        Mide la diversidad considerando tanto el número de categorías
        como la uniformidad de su distribución.
        
        Args:
            series (pd.Series): Serie con categorías
            
        Returns:
            float: Índice de diversidad de Shannon
        """
        value_counts = series.value_counts()
        proportions = value_counts / len(series)
        return -sum(proportions * np.log(proportions))
    
    def simpson_diversity(series):
        """
        Calcula el índice de diversidad de Simpson.
        
        Mide la probabilidad de que dos elementos seleccionados
        aleatoriamente pertenezcan a categorías diferentes.
        
        Args:
            series (pd.Series): Serie con categorías
            
        Returns:
            float: Índice de diversidad de Simpson
        """
        value_counts = series.value_counts()
        proportions = value_counts / len(series)
        return 1 - sum(proportions**2)
    
    def gini_index(series):
        """
        Calcula el índice de Gini (concentración).
        
        Mide la desigualdad en la distribución de categorías.
        Valores más altos indican mayor concentración.
        
        Args:
            series (pd.Series): Serie con categorías
            
        Returns:
            float: Índice de Gini
        """
        value_counts = series.value_counts()
        n = len(series)
        cumsum = np.cumsum(sorted(value_counts))
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * (n + 1))
    
    # === ANÁLISIS POR DÉCADA ===
    diversity_metrics = {}
    for decade in ['2000s', '2010s']:
        decade_data = final_df[final_df['decade'] == decade]['genre']
        diversity_metrics[f'{decade}_shannon'] = shannon_diversity(decade_data)
        diversity_metrics[f'{decade}_simpson'] = simpson_diversity(decade_data)
        diversity_metrics[f'{decade}_gini'] = gini_index(decade_data)
        diversity_metrics[f'{decade}_unique_genres'] = decade_data.nunique()
        diversity_metrics[f'{decade}_total_movies'] = len(decade_data)
    
    # === CREAR DATAFRAME DE COMPARACIÓN ===
    comparison_df = pd.DataFrame({
        'Métrica': ['Shannon Diversity', 'Simpson Diversity', 'Gini Index', 'Géneros Únicos', 'Total Películas'],
        '2000s': [
            diversity_metrics['2000s_shannon'],
            diversity_metrics['2000s_simpson'],
            diversity_metrics['2000s_gini'],
            diversity_metrics['2000s_unique_genres'],
            diversity_metrics['2000s_total_movies']
        ],
        '2010s': [
            diversity_metrics['2010s_shannon'],
            diversity_metrics['2010s_simpson'],
            diversity_metrics['2010s_gini'],
            diversity_metrics['2010s_unique_genres'],
            diversity_metrics['2010s_total_movies']
        ]
    })
    
    # Calcular cambios porcentuales
    comparison_df['Cambio'] = ((comparison_df['2010s'] - comparison_df['2000s']) / comparison_df['2000s'] * 100).round(2)
    print("\nComparación de Métricas de Diversidad:")
    display(comparison_df)
    
    return diversity_metrics


def analyze_genre_associations(final_df):
    """
    Análisis de co-ocurrencia y asociación entre géneros.
    
    Esta función analiza las asociaciones entre géneros cinematográficos mediante:
    - Matriz de co-ocurrencia de géneros por película
    - Matriz de correlación entre géneros
    - Análisis de clústeres usando K-means
    - Identificación de géneros que frecuentemente aparecen juntos
    
    Args:
        final_df (pd.DataFrame): Dataset final con datos integrados
        
    Returns:
        dict: Resultados del análisis de asociaciones
    """
    print("=== ANÁLISIS DE ASOCIACIÓN ENTRE GÉNEROS ===")
    
    # === 1. MATRIZ DE CO-OCURRENCIA DE GÉNEROS POR PELÍCULA ===
    genre_pivot = final_df.pivot_table(
        index='id', 
        columns='genre', 
        values='decade', 
        aggfunc='count', 
        fill_value=0
    )
    
    # Convertir a binario (género presente/ausente)
    genre_binary = (genre_pivot > 0).astype(int)
    
    # === 2. MATRIZ DE CORRELACIÓN ENTRE GÉNEROS ===
    correlation_matrix = genre_binary.corr()
    
    print("Top 10 Correlaciones Positivas entre Géneros:")
    # Obtener pares de géneros con mayor correlación
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            genre1 = correlation_matrix.columns[i]
            genre2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            corr_pairs.append((genre1, genre2, corr_value))
    
    corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)
    top_correlations = pd.DataFrame(corr_pairs[:10], columns=['Género 1', 'Género 2', 'Correlación'])
    display(top_correlations.round(4))
    
    # === 3. ANÁLISIS DE CLÚSTERES DE GÉNEROS ===
    # Usar solo géneros con suficiente frecuencia para clustering
    genre_freq = final_df['genre'].value_counts()
    frequent_genres = genre_freq[genre_freq >= 100].index
    genre_subset = genre_binary[frequent_genres]
    
    # Normalizar datos para clustering
    scaler = StandardScaler()
    genre_scaled = scaler.fit_transform(genre_subset.T)
    
    # K-means clustering con selección automática de k
    best_k = 3
    best_score = -1
    for k in range(2, min(8, len(frequent_genres))):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(genre_scaled)
        score = silhouette_score(genre_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    # Aplicar clustering con el mejor k
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(genre_scaled)
    
    # Crear DataFrame de clusters
    genre_clusters = pd.DataFrame({
        'Género': frequent_genres,
        'Cluster': cluster_labels
    })
    
    print(f"\nAnálisis de Clústeres (k={best_k}, Silhouette Score: {best_score:.3f}):")
    for cluster in range(best_k):
        cluster_genres = genre_clusters[genre_clusters['Cluster'] == cluster]['Género'].tolist()
        print(f"Cluster {cluster}: {cluster_genres}")
    
    return {
        'correlation_matrix': correlation_matrix,
        'genre_clusters': genre_clusters,
        'top_correlations': top_correlations
    }


def analyze_categorical_trends(final_df):
    """
    Análisis de tendencias temporales en variables categóricas.
    
    Esta función analiza las tendencias temporales de géneros cinematográficos:
    - Tendencias por año
    - Identificación de géneros emergentes/declinantes
    - Análisis de estacionalidad
    - Cálculo de pendientes de crecimiento/declive
    
    Args:
        final_df (pd.DataFrame): Dataset final con datos integrados
        
    Returns:
        dict: Resultados del análisis de tendencias temporales
    """
    print("=== ANÁLISIS DE TENDENCIAS TEMPORALES CATEGÓRICAS ===")
    
    # === PREPARACIÓN DE DATOS TEMPORALES ===
    # Agregar año si no existe
    if 'first_date' in final_df.columns:
        final_df['year'] = pd.to_datetime(final_df['first_date']).dt.year
    else:
        # Crear año simulado basado en década
        final_df['year'] = final_df['decade'].map({'2000s': 2005, '2010s': 2015})
    
    # === 1. ANÁLISIS DE TENDENCIAS POR AÑO ===
    yearly_genres = final_df.groupby(['year', 'genre']).size().unstack(fill_value=0)
    
    # === 2. IDENTIFICAR GÉNEROS EMERGENTES/DECLINANTES ===
    # Calcular tendencia lineal para cada género
    genre_trends = {}
    for genre in yearly_genres.columns:
        if genre in yearly_genres.columns:
            y_values = yearly_genres[genre].values
            x_values = np.arange(len(y_values))
            if len(y_values) > 1 and np.sum(y_values) > 0:
                # Calcular pendiente de la línea de tendencia
                slope = np.polyfit(x_values, y_values, 1)[0]
                genre_trends[genre] = slope
    
    # Ordenar por tendencia
    trend_df = pd.DataFrame(list(genre_trends.items()), columns=['Género', 'Tendencia'])
    trend_df = trend_df.sort_values('Tendencia', ascending=False)
    
    print("Géneros con Mayor Crecimiento Temporal:")
    display(trend_df.head(10))
    
    print("\nGéneros con Mayor Declive Temporal:")
    display(trend_df.tail(10))
    
    # === 3. ANÁLISIS DE ESTACIONALIDAD EN GÉNEROS ===
    if 'first_date' in final_df.columns:
        final_df['month'] = pd.to_datetime(final_df['first_date']).dt.month
        seasonal_genres = final_df.groupby(['month', 'genre']).size().unstack(fill_value=0)
        
        print("\nAnálisis de Estacionalidad (Top 5 géneros por mes):")
        for month in range(1, 13):
            month_data = seasonal_genres.loc[month].nlargest(5)
            print(f"Mes {month}: {month_data.to_dict()}")
    else:
        seasonal_genres = None
    
    return {
        'yearly_trends': yearly_genres,
        'genre_trends': trend_df,
        'seasonal_patterns': seasonal_genres
    }


def create_advanced_categorical_visualizations(final_df):
    """
    Visualizaciones avanzadas para análisis categórico.
    
    Esta función genera 12 gráficos avanzados para análisis de variables categóricas:
    1. Mosaic plot de géneros vs décadas
    2. Heatmap de cambios porcentuales
    3. Gráfico de barras apiladas por década
    4. Análisis de diversidad temporal
    5. Boxplot de distribución de géneros por película
    6. Análisis de concentración (Top géneros)
    7. Matriz de correlación entre géneros
    8. Análisis de transiciones de géneros
    9. Análisis de estabilidad de rankings
    10. Segmentación de géneros por participación
    11. Evolución de la diversidad temporal
    12. Resumen estadístico
    
    Args:
        final_df (pd.DataFrame): Dataset final con datos integrados
        
    Returns:
        dict: Confirmación de visualizaciones completadas
    """
    print("=== CREANDO VISUALIZACIONES AVANZADAS CATEGÓRICAS ===")
    
    # === CONFIGURACIÓN DE GRÁFICOS ===
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (20, 24)
    plt.rcParams['font.size'] = 10
    
    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(4, 3, figsize=(24, 20))
    fig.suptitle('Análisis Avanzado de Variables Categóricas - Géneros Cinematográficos', 
                 fontsize=16, fontweight='bold')
    
    # === 1. MOSAIC PLOT DE GÉNEROS VS DÉCADAS ===
    try:
        from statsmodels.graphics.mosaicplot import mosaic
        mosaic_data = final_df.groupby(['decade', 'genre']).size()
        mosaic(mosaic_data, ax=axes[0,0], title='Distribución de Géneros por Década')
    except ImportError:
        # Fallback si no está disponible statsmodels
        decade_genre_counts = final_df.groupby(['decade', 'genre']).size().unstack(fill_value=0)
        decade_genre_counts.plot(kind='bar', stacked=True, ax=axes[0,0], 
                                title='Distribución de Géneros por Década')
    
    # === 2. HEATMAP DE CAMBIOS PORCENTUALES ===
    proportions = pd.crosstab(final_df['genre'], final_df['decade'], normalize='index')
    changes = ((proportions['2010s'] - proportions['2000s']) / proportions['2000s'] * 100).dropna()
    changes_df = pd.DataFrame({'Cambio %': changes.values}, index=changes.index)
    
    sns.heatmap(changes_df.values, 
                yticklabels=changes_df.index, 
                xticklabels=['Cambio %'],
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=axes[0,1])
    axes[0,1].set_title('Cambios Porcentuales por Género')
    
    # === 3. GRÁFICO DE BARRAS APILADAS POR DÉCADA ===
    decade_genre_counts = final_df.groupby(['decade', 'genre']).size().unstack(fill_value=0)
    top_10_genres = final_df['genre'].value_counts().head(10).index
    decade_genre_counts[top_10_genres].plot(kind='bar', stacked=True, ax=axes[0,2])
    axes[0,2].set_title('Top 10 Géneros por Década (Apilado)')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # === 4. ANÁLISIS DE DIVERSIDAD TEMPORAL ===
    def shannon_diversity(series):
        """Función auxiliar para calcular diversidad de Shannon."""
        value_counts = series.value_counts()
        proportions = value_counts / len(series)
        return -sum(proportions * np.log(proportions))
    
    # Calcular diversidad por año si es posible
    if 'first_date' in final_df.columns:
        final_df['year'] = pd.to_datetime(final_df['first_date']).dt.year
        diversity_by_year = final_df.groupby('year')['genre'].apply(shannon_diversity)
        axes[1,0].plot(diversity_by_year.index, diversity_by_year.values, marker='o')
        axes[1,0].set_title('Evolución de la Diversidad de Géneros')
        axes[1,0].set_xlabel('Año')
        axes[1,0].set_ylabel('Índice de Shannon')
        axes[1,0].grid(True, alpha=0.3)
    else:
        # Análisis por década
        diversity_2000s = shannon_diversity(final_df[final_df['decade'] == '2000s']['genre'])
        diversity_2010s = shannon_diversity(final_df[final_df['decade'] == '2010s']['genre'])
        decades = ['2000s', '2010s']
        diversities = [diversity_2000s, diversity_2010s]
        axes[1,0].bar(decades, diversities, color=['lightblue', 'lightcoral'])
        axes[1,0].set_title('Diversidad de Géneros por Década')
        axes[1,0].set_ylabel('Índice de Shannon')
    
    # === 5. BOXPLOT DE DISTRIBUCIÓN DE GÉNEROS POR PELÍCULA ===
    movies_genre_count = final_df.groupby(['id', 'decade']).size()
    movies_genre_count.unstack().boxplot(ax=axes[1,1])
    axes[1,1].set_title('Distribución de Géneros por Película')
    axes[1,1].set_ylabel('Número de Géneros')
    
    # === 6. ANÁLISIS DE CONCENTRACIÓN (TOP GÉNEROS) ===
    top_genres = final_df['genre'].value_counts().head(15)
    axes[1,2].pie(top_genres.values, labels=top_genres.index, 
                  autopct='%1.1f%%', startangle=90)
    axes[1,2].set_title('Concentración de Géneros (Top 15)')
    
    # === 7. ANÁLISIS DE CORRELACIÓN ENTRE GÉNEROS ===
    genre_pivot = final_df.pivot_table(index='id', columns='genre', values='decade', 
                                      aggfunc='count', fill_value=0)
    genre_binary = (genre_pivot > 0).astype(int)
    correlation_matrix = genre_binary.corr()
    
    # Mostrar solo correlaciones significativas
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, ax=axes[2,0])
    axes[2,0].set_title('Matriz de Correlación entre Géneros')
    
    # === 8. ANÁLISIS DE TRANSICIONES DE GÉNEROS ===
    genre_changes = ((proportions['2010s'] - proportions['2000s']) / proportions['2000s'] * 100).dropna()
    colors = ['green' if x > 0 else 'red' for x in genre_changes.values]
    bars = axes[2,1].barh(range(len(genre_changes)), genre_changes.values, 
                          color=colors, alpha=0.7)
    axes[2,1].set_yticks(range(len(genre_changes)))
    axes[2,1].set_yticklabels(genre_changes.index)
    axes[2,1].set_title('Cambios en Popularidad de Géneros')
    axes[2,1].set_xlabel('Cambio Porcentual (%)')
    axes[2,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # === 9. ANÁLISIS DE ESTABILIDAD DE RANKINGS ===
    top_5_2000s = final_df[final_df['decade'] == '2000s']['genre'].value_counts().head(5)
    top_5_2010s = final_df[final_df['decade'] == '2010s']['genre'].value_counts().head(5)
    
    # Crear DataFrame comparativo
    comparison_genres = set(top_5_2000s.index) | set(top_5_2010s.index)
    comparison_data = []
    for genre in comparison_genres:
        rank_2000s = list(top_5_2000s.index).index(genre) + 1 if genre in top_5_2000s.index else 6
        rank_2010s = list(top_5_2010s.index).index(genre) + 1 if genre in top_5_2010s.index else 6
        comparison_data.append({'Género': genre, 'Ranking 2000s': rank_2000s, 'Ranking 2010s': rank_2010s})
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Cambio Ranking'] = comparison_df['Ranking 2000s'] - comparison_df['Ranking 2010s']
    
    # Gráfico de cambio de rankings
    x = np.arange(len(comparison_df))
    axes[2,2].scatter(comparison_df['Ranking 2000s'], comparison_df['Ranking 2010s'], 
                     s=100, alpha=0.7, c=comparison_df['Cambio Ranking'], cmap='RdYlGn')
    for i, genre in enumerate(comparison_df['Género']):
        axes[2,2].annotate(genre, (comparison_df['Ranking 2000s'].iloc[i], comparison_df['Ranking 2010s'].iloc[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[2,2].set_xlabel('Ranking 2000s')
    axes[2,2].set_ylabel('Ranking 2010s')
    axes[2,2].set_title('Estabilidad de Rankings de Géneros')
    axes[2,2].plot([1, 6], [1, 6], 'k--', alpha=0.5)  # Línea de igualdad
    
    # === 10. ANÁLISIS DE SEGMENTACIÓN DE GÉNEROS ===
    genre_counts = final_df['genre'].value_counts()
    total_movies = len(final_df['id'].unique())
    
    # Clasificar géneros por participación en el mercado
    dominant_genres = genre_counts[genre_counts / total_movies > 0.05]
    niche_genres = genre_counts[genre_counts / total_movies < 0.01]
    emerging_genres = genre_counts[(genre_counts / total_movies >= 0.01) & (genre_counts / total_movies <= 0.05)]
    
    segment_counts = {
        'Dominantes (>5%)': len(dominant_genres),
        'Emergentes (1-5%)': len(emerging_genres),
        'Nicho (<1%)': len(niche_genres)
    }
    
    axes[3,0].pie(segment_counts.values(), labels=segment_counts.keys(), 
                  autopct='%1.1f%%', startangle=90)
    axes[3,0].set_title('Segmentación de Géneros por Participación')
    
    # === 11. ANÁLISIS DE DIVERSIDAD TEMPORAL (SI HAY DATOS DE FECHA) ===
    if 'first_date' in final_df.columns:
        final_df['year'] = pd.to_datetime(final_df['first_date']).dt.year
        yearly_diversity = final_df.groupby('year')['genre'].apply(shannon_diversity)
        axes[3,1].plot(yearly_diversity.index, yearly_diversity.values, 
                       marker='o', linewidth=2)
        axes[3,1].set_title('Evolución de la Diversidad Temporal')
        axes[3,1].set_xlabel('Año')
        axes[3,1].set_ylabel('Índice de Shannon')
        axes[3,1].grid(True, alpha=0.3)
    else:
        # Análisis de concentración por década
        concentration_2000s = 1 - sum((final_df[final_df['decade'] == '2000s']['genre'].value_counts() / len(final_df[final_df['decade'] == '2000s']))**2)
        concentration_2010s = 1 - sum((final_df[final_df['decade'] == '2010s']['genre'].value_counts() / len(final_df[final_df['decade'] == '2010s']))**2)
        
        decades = ['2000s', '2010s']
        concentrations = [concentration_2000s, concentration_2010s]
        axes[3,1].bar(decades, concentrations, color=['lightblue', 'lightcoral'])
        axes[3,1].set_title('Índice de Simpson por Década')
        axes[3,1].set_ylabel('Índice de Simpson')
    
    # === 12. RESUMEN ESTADÍSTICO ===
    stats_text = f"""
    RESUMEN ESTADÍSTICO:
    
    Total de Géneros: {final_df['genre'].nunique()}
    Total de Películas: {final_df['id'].nunique():,}
    
    Géneros Dominantes: {len(dominant_genres)}
    Géneros Emergentes: {len(emerging_genres)}
    Géneros de Nicho: {len(niche_genres)}
    
    Diversidad 2000s: {shannon_diversity(final_df[final_df['decade'] == '2000s']['genre']):.3f}
    Diversidad 2010s: {shannon_diversity(final_df[final_df['decade'] == '2010s']['genre']):.3f}
    """
    
    axes[3,2].text(0.1, 0.5, stats_text, transform=axes[3,2].transAxes, fontsize=10,
                   verticalalignment='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    axes[3,2].set_xlim(0, 1)
    axes[3,2].set_ylim(0, 1)
    axes[3,2].axis('off')
    axes[3,2].set_title('Resumen Estadístico')
    
    # === GUARDAR Y MOSTRAR GRÁFICOS ===
    plt.tight_layout()
    plt.savefig('data/08_reporting/advanced_categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {"advanced_categorical_visualizations": "completed"}


def calculate_advanced_categorical_metrics(final_df):
    """
    Métricas avanzadas para análisis categórico.
    
    Esta función calcula métricas estadísticas avanzadas para comparar
    la distribución de géneros entre décadas, incluyendo:
    - Entropía de Shannon
    - Índice de Gini
    - Índice de Herfindahl-Hirschman
    - Test de Chi-cuadrado para independencia
    - Cramér's V para efecto tamaño
    
    Args:
        final_df (pd.DataFrame): Dataset final con datos integrados
        
    Returns:
        dict: Métricas avanzadas y resultados de tests estadísticos
    """
    print("=== MÉTRICAS AVANZADAS CATEGÓRICAS ===")
    
    def entropy(series):
        """
        Calcula la entropía de Shannon.
        
        Mide la incertidumbre promedio en la distribución de categorías.
        Valores más altos indican mayor diversidad.
        
        Args:
            series (pd.Series): Serie con categorías
            
        Returns:
            float: Entropía de Shannon
        """
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        return -sum(probabilities * np.log2(probabilities))
    
    def gini_index(series):
        """
        Calcula el índice de Gini.
        
        Mide la desigualdad en la distribución de categorías.
        Valores más altos indican mayor concentración.
        
        Args:
            series (pd.Series): Serie con categorías
            
        Returns:
            float: Índice de Gini
        """
        value_counts = series.value_counts()
        n = len(series)
        cumsum = np.cumsum(sorted(value_counts))
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * (n + 1))
    
    def herfindahl_index(series):
        """
        Calcula el índice de Herfindahl-Hirschman.
        
        Mide la concentración del mercado. Valores más altos
        indican mayor concentración en pocas categorías.
        
        Args:
            series (pd.Series): Serie con categorías
            
        Returns:
            float: Índice de Herfindahl-Hirschman
        """
        proportions = series.value_counts() / len(series)
        return sum(proportions**2)
    
    # === CALCULAR MÉTRICAS POR DÉCADA ===
    metrics = {}
    for decade in ['2000s', '2010s']:
        decade_data = final_df[final_df['decade'] == decade]['genre']
        metrics[f'{decade}_entropy'] = entropy(decade_data)
        metrics[f'{decade}_gini'] = gini_index(decade_data)
        metrics[f'{decade}_herfindahl'] = herfindahl_index(decade_data)
        metrics[f'{decade}_unique_genres'] = decade_data.nunique()
        metrics[f'{decade}_total_movies'] = len(decade_data)
        metrics[f'{decade}_most_common'] = decade_data.value_counts().index[0]
        metrics[f'{decade}_most_common_pct'] = (decade_data.value_counts().iloc[0] / len(decade_data)) * 100
    
    # === CREAR DATAFRAME DE COMPARACIÓN ===
    comparison_df = pd.DataFrame({
        'Métrica': [
            'Entropía de Shannon', 'Índice de Gini', 'Índice de Herfindahl',
            'Géneros Únicos', 'Total Películas', 'Género Más Común (%)'
        ],
        '2000s': [
            metrics['2000s_entropy'],
            metrics['2000s_gini'],
            metrics['2000s_herfindahl'],
            metrics['2000s_unique_genres'],
            metrics['2000s_total_movies'],
            metrics['2000s_most_common_pct']
        ],
        '2010s': [
            metrics['2010s_entropy'],
            metrics['2010s_gini'],
            metrics['2010s_herfindahl'],
            metrics['2010s_unique_genres'],
            metrics['2010s_total_movies'],
            metrics['2010s_most_common_pct']
        ]
    })
    
    # Calcular cambios porcentuales
    comparison_df['Cambio %'] = ((comparison_df['2010s'] - comparison_df['2000s']) / comparison_df['2000s'] * 100).round(2)
    print("\nComparación de Métricas Avanzadas:")
    display(comparison_df)
    
    # === TEST DE SIGNIFICANCIA ESTADÍSTICA ===
    contingency = pd.crosstab(final_df['decade'], final_df['genre'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"\nTest de Independencia Chi-cuadrado:")
    print(f"Chi2: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Grados de libertad: {dof}")
    print(f"Significativo (p<0.05): {'Sí' if p_value < 0.05 else 'No'}")
    
    # === ANÁLISIS DE EFECTO TAMAÑO (CRAMÉR'S V) ===
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    print(f"Cramér's V: {cramers_v:.4f}")
    
    return {
        'metrics': metrics,
        'comparison': comparison_df,
        'chi2_test': {'chi2': chi2, 'p_value': p_value, 'dof': dof},
        'cramers_v': cramers_v
    }


# =============================================================================
# FUNCIÓN PRINCIPAL DEL PIPELINE
# =============================================================================

def create_eda_pipeline() -> Pipeline:
    """
    Crea el pipeline de EDA completo con análisis categóricos avanzados.
    
    Este pipeline incluye:
    
    ANÁLISIS BÁSICO:
    - Carga y visualización de datasets
    - Medidas de tendencia central
    - Medidas de dispersión
    - Medidas de posición
    - Visualizaciones de calidad de datos
    - Análisis temporal
    - Análisis de géneros
    
    ANÁLISIS CATEGÓRICOS AVANZADOS:
    - Distribución de géneros por década
    - Análisis de diversidad y concentración
    - Asociaciones entre géneros
    - Tendencias temporales categóricas
    - Visualizaciones avanzadas (12 gráficos)
    - Métricas estadísticas avanzadas
    
    Returns:
        Pipeline: Pipeline de Kedro con todos los nodos de análisis
    """
    return pipeline([
        # === ANÁLISIS BÁSICO (EXISTENTE) ===
        node(
            func=load_and_display_datasets,
            inputs=["releases", "genres", "countries"],
            outputs="eda_basic_info",
            name="load_and_display_datasets"
        ),
        node(
            func=calculate_central_tendency,
            inputs=["releases", "genres", "countries"],
            outputs="eda_central_tendency",
            name="calculate_central_tendency"
        ),
        node(
            func=calculate_dispersion,
            inputs=["releases", "genres", "countries"],
            outputs="eda_dispersion",
            name="calculate_dispersion"
        ),
        node(
            func=calculate_position_measures,
            inputs=["releases", "genres", "countries"],
            outputs="eda_position_measures",
            name="calculate_position_measures"
        ),
        node(
            func=create_quality_visualizations,
            inputs=["releases", "genres", "countries"],
            outputs="eda_quality_visualizations",
            name="create_quality_visualizations"
        ),
        node(
            func=create_temporal_analysis,
            inputs=["releases"],
            outputs="eda_temporal_analysis",
            name="create_temporal_analysis"
        ),
        node(
            func=create_genre_analysis,
            inputs=["genres"],
            outputs="eda_genre_analysis",
            name="create_genre_analysis"
        ),
        
        # === ANÁLISIS CATEGÓRICOS AVANZADOS (NUEVOS) ===
        node(
            func=analyze_genre_distribution_by_decade,
            inputs=["final_df"],
            outputs="eda_genre_distribution",
            name="analyze_genre_distribution_by_decade"
        ),
        node(
            func=analyze_genre_diversity,
            inputs=["final_df"],
            outputs="eda_genre_diversity",
            name="analyze_genre_diversity"
        ),
        node(
            func=analyze_genre_associations,
            inputs=["final_df"],
            outputs="eda_genre_associations",
            name="analyze_genre_associations"
        ),
        node(
            func=analyze_categorical_trends,
            inputs=["final_df"],
            outputs="eda_categorical_trends",
            name="analyze_categorical_trends"
        ),
        node(
            func=create_advanced_categorical_visualizations,
            inputs=["final_df"],
            outputs="eda_advanced_categorical_visualizations",
            name="create_advanced_categorical_visualizations"
        ),
        node(
            func=calculate_advanced_categorical_metrics,
            inputs=["final_df"],
            outputs="eda_advanced_categorical_metrics",
            name="calculate_advanced_categorical_metrics"
        ),
        
        # === PREPARACIÓN DE DATASETS PARA ML ===
        node(
            func=prepare_classification_data,
            inputs=["releases", "countries", "genres", "movies"],
            outputs="classification_dataset",
            name="prepare_classification_data"
        ),
        node(
            func=prepare_regression_data,
            inputs=["movies", "genres"],
            outputs="regression_dataset",
            name="prepare_regression_data"
        )
    ])
