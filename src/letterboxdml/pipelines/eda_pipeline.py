"""Pipeline para Fase 2: Análisis Exploratorio de Datos (EDA)."""
from kedro.pipeline import Pipeline, node, pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


def load_and_display_datasets(releases, genres, countries):
    """Carga y muestra información básica de los datasets."""
    datasets = {
        "releases": releases,
        "genres": genres, 
        "countries": countries
    }
    
    results = {}
    for name, df in datasets.items():
        print(f"\n=== DATASET: {name} ===")
        print("shape:", df.shape)
        
        # Mostrar primeras 5 filas
        display(df.head(5))
        
        # Mostrar tipos de datos
        schema = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str)
        })
        print("dtypes por columna:")
        display(schema)
        
        results[f"{name}_info"] = {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "sample": df.head(5)
        }
    
    return results


def calculate_central_tendency(releases, genres, countries):
    """Calcula medidas de tendencia central para cada dataset."""
    def central_tendency_by_dataset(df, dataset_name, date_col=None):
        print("="*40)
        print(f"\n=== Medidas de tendencia central — {dataset_name} ===")
        print("="*40)
        
        # Numéricas: media y mediana (excluye 'id')
        num = df.select_dtypes(include=[np.number]).drop(columns=["id"], errors="ignore")
        if num.shape[1] > 0:
            out_num = pd.DataFrame({
                "mean": num.mean(numeric_only=True),
                "median": num.median(numeric_only=True)
            })
            print("Numéricas (media, mediana):")
            display(out_num.round(4))
        else:
            print("Numéricas: (sin columnas numéricas relevantes)")
        
        # Categóricas: moda (valor más frecuente)
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            rows, n = [], len(df)
            for c in cat_cols:
                vc = df[c].value_counts(dropna=False)
                mode_val = vc.index[0] if len(vc) else None
                mode_freq = int(vc.iloc[0]) if len(vc) else 0
                mode_share = (mode_freq / n) if n else np.nan
                rows.append({"column": c, "mode": mode_val, "mode_freq": mode_freq, "mode_share": round(mode_share, 4)})
            print("Categóricas (moda, frecuencia, participación):")
            display(pd.DataFrame(rows).sort_values("mode_share", ascending=False))
        else:
            print("Categóricas: (sin columnas categóricas)")
        
        # Fechas: mediana (si aplica)
        if date_col and date_col in df.columns:
            s = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
            med = s.median() if s.notna().any() else None
            print(f"Mediana de fecha ({date_col}): {med}")
    
    # Aplicar análisis a cada dataset
    central_tendency_by_dataset(releases, "releases", date_col="date")
    central_tendency_by_dataset(genres, "genres")
    central_tendency_by_dataset(countries, "countries")
    
    return {"central_tendency_analysis": "completed"}


def calculate_dispersion(releases, genres, countries):
    """Calcula medidas de dispersión para cada dataset."""
    def _numeric(df):
        """Selecciona columnas numéricas relevantes (excluye 'id')."""
        return df.select_dtypes(include=[np.number]).drop(columns=["id"], errors="ignore")

    def _metrics(series: pd.Series) -> pd.Series:
        """Calcula var, std, rango, coef. variación, IQR para una serie numérica."""
        s = series.dropna()
        if s.empty:
            return pd.Series({"var": np.nan, "std": np.nan, "range": np.nan, "cv": np.nan, "iqr": np.nan})
        var = s.var()  # ddof=1 (muestra)
        std = s.std()  # ddof=1
        rng = s.max() - s.min()
        mean = s.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = std / mean if mean not in (0, np.nan) else np.nan
        q75, q25 = s.quantile(0.75), s.quantile(0.25)
        iqr = q75 - q25
        return pd.Series({"var": var, "std": std, "range": rng, "cv": cv, "iqr": iqr})

    def dispersion_table(df: pd.DataFrame) -> pd.DataFrame:
        """Tabla de dispersión para todas las columnas numéricas (excluyendo id)."""
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
        """Convierte una columna de fecha a días (float) para calcular dispersión temporal."""
        if date_col not in df.columns:
            return None
        s = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None).dropna()
        if s.empty:
            return None
        # ns -> s -> días (uso astype para evitar FutureWarning)
        days = (s.astype("int64") / 1e9) / 86400.0
        return pd.Series(days, index=s.index)

    print("="*40)
    print("=== DISPERSIÓN — releases (numéricas) ===")
    print("="*40)
    display(dispersion_table(releases))

    # Extra: dispersión temporal de 'date' en días
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
    """Calcula medidas de posición para cada dataset."""
    def _numeric(df):
        """Selecciona columnas numéricas relevantes (excluye 'id')."""
        return df.select_dtypes(include=[np.number]).drop(columns=["id"], errors="ignore")

    def numeric_position_table(df: pd.DataFrame) -> pd.DataFrame:
        num = _numeric(df)
        if num.shape[1] == 0:
            return pd.DataFrame({"note": ["(sin columnas numéricas relevantes)"]})
        desc = num.describe(percentiles=[.10, .25, .50, .75, .90]).T
        # Renombrar percentiles para claridad y agregar suma total
        desc = desc.rename(columns={"count":"count", "mean":"mean", "std":"std", "min":"min",
                                   "10%":"p10", "25%":"p25", "50%":"median", "75%":"p75", "90%":"p90", "max":"max"})
        desc["sum"] = num.sum()
        # Ordenar columnas típicas de posición
        cols = ["count", "sum", "min", "p10", "p25", "median", "p75", "p90", "max"]
        return desc[cols].round(6)

    def date_position_table(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if date_col not in df.columns:
            return pd.DataFrame({"note": [f"(columna '{date_col}' no existe)"]})
        s = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None).dropna()
        if s.empty:
            return pd.DataFrame({"note": [f"(no parseable '{date_col}')"]})
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


def create_quality_visualizations(releases, genres, countries):
    """Crea visualizaciones de calidad de datos."""
    # Configuración de gráficos
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10

    # Crear figura con subplots para análisis de calidad
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Análisis de Calidad de Datos - Valores Faltantes y Completitud', fontsize=16, fontweight='bold')

    # 1. Valores faltantes por dataset
    datasets = ['releases', 'genres', 'countries']
    missing_data = []

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

    # 2. Heatmap de completitud
    completeness_pivot = 100 - missing_pivot
    sns.heatmap(completeness_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=axes[0,1], cbar_kws={'label': 'Completitud (%)'})
    axes[0,1].set_title('Mapa de Calor - Completitud por Dataset y Columna')
    axes[0,1].set_xlabel('Dataset')
    axes[0,1].set_ylabel('Columna')

    # 3. Distribución de tipos de lanzamiento (releases.type)
    type_counts = releases['type'].value_counts()
    axes[0,2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0,2].set_title('Distribución de Tipos de Lanzamiento')

    # 4. Top 10 países por número de estrenos
    country_counts = releases['country'].value_counts().head(10)
    axes[1,0].barh(range(len(country_counts)), country_counts.values, color='lightblue', alpha=0.7)
    axes[1,0].set_yticks(range(len(country_counts)))
    axes[1,0].set_yticklabels(country_counts.index)
    axes[1,0].set_title('Top 10 Países por Número de Estrenos')
    axes[1,0].set_xlabel('Número de Estrenos')
    axes[1,0].grid(True, alpha=0.3)

    # 5. Análisis de outliers temporales
    releases_temp = releases.copy()
    releases_temp['date_parsed'] = pd.to_datetime(releases_temp['date'], errors='coerce')
    releases_temp = releases_temp.dropna(subset=['date_parsed'])
    releases_temp['year'] = releases_temp['date_parsed'].dt.year

    # Filtrar años razonables para el análisis de outliers
    yearly_counts = releases_temp.groupby('year').size()
    axes[1,1].plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=1, markersize=3)
    axes[1,1].set_title('Distribución de Estrenos por Año (Análisis de Outliers)')
    axes[1,1].set_xlabel('Año')
    axes[1,1].set_ylabel('Número de Estrenos')
    axes[1,1].grid(True, alpha=0.3)

    # 6. Distribución de géneros (top 15)
    genre_counts = genres['genre'].value_counts().head(15)
    axes[1,2].bar(range(len(genre_counts)), genre_counts.values, color='lightgreen', alpha=0.7)
    axes[1,2].set_xticks(range(len(genre_counts)))
    axes[1,2].set_xticklabels(genre_counts.index, rotation=45, ha='right')
    axes[1,2].set_title('Top 15 Géneros Más Frecuentes')
    axes[1,2].set_ylabel('Número de Películas')
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/08_reporting/quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Resumen estadístico de calidad
    print("=== RESUMEN DE CALIDAD DE DATOS ===")
    print(f"Total de registros en releases: {len(releases):,}")
    print(f"Total de registros en genres: {len(genres):,}")
    print(f"Total de registros en countries: {len(countries):,}")
    print(f"\nValores faltantes en rating: {releases['rating'].isnull().sum():,} ({releases['rating'].isnull().mean()*100:.1f}%)")
    print(f"Rango temporal: {releases_temp['year'].min()}-{releases_temp['year'].max()}")

    return {"quality_visualizations": "completed"}


def create_temporal_analysis(releases):
    """Crea análisis temporal y visualizaciones."""
    # Configuración de gráficos
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (18, 12)
    plt.rcParams['font.size'] = 10

    # Preparar datos temporales
    releases_temp = releases.copy()
    releases_temp['date_parsed'] = pd.to_datetime(releases_temp['date'], errors='coerce')
    releases_temp = releases_temp.dropna(subset=['date_parsed'])
    releases_temp['year'] = releases_temp['date_parsed'].dt.year
    releases_temp['month'] = releases_temp['date_parsed'].dt.month
    releases_temp['quarter'] = releases_temp['date_parsed'].dt.quarter
    releases_temp['day_of_week'] = releases_temp['date_parsed'].dt.day_name()

    # Filtrar años razonables (1900-2025) para eliminar outliers temporales
    releases_temp = releases_temp[(releases_temp['year'] >= 1900) & (releases_temp['year'] <= 2025)]

    # Crear figura con subplots para análisis temporal
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('Análisis de Tendencias Temporales y Patrones Estacionales', fontsize=16, fontweight='bold')

    # 1. Tendencia anual (2000-2020 para mejor visualización)
    releases_2000_2020 = releases_temp[(releases_temp['year'] >= 2000) & (releases_temp['year'] <= 2020)]
    yearly_counts = releases_2000_2020.groupby('year').size()

    axes[0,0].plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=6, color='blue')
    axes[0,0].set_title('Tendencia de Estrenos Anuales (2000-2020)')
    axes[0,0].set_xlabel('Año')
    axes[0,0].set_ylabel('Número de Estrenos')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Distribución por mes (todos los años)
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

    # 3. Distribución por trimestre
    quarterly_counts = releases_temp.groupby('quarter').size()
    quarter_labels = ['Q1 (Ene-Mar)', 'Q2 (Abr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dic)']
    axes[1,0].pie(quarterly_counts.values, labels=quarter_labels, autopct='%1.1f%%', startangle=90, 
                  colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    axes[1,0].set_title('Distribución de Estrenos por Trimestre')

    # 4. Distribución por día de la semana
    dow_counts = releases_temp.groupby('day_of_week').size()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = dow_counts.reindex(dow_order)
    dow_spanish = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    bars = axes[1,1].bar(range(len(dow_counts)), dow_counts.values, color='lightgreen', alpha=0.7)
    axes[1,1].set_title('Distribución de Estrenos por Día de la Semana')
    axes[1,1].set_xlabel('Día de la Semana')
    axes[1,1].set_ylabel('Número de Estrenos')
    axes[1,1].set_xticks(range(len(dow_counts)))
    axes[1,1].set_xticklabels(dow_spanish, rotation=45)
    axes[1,1].grid(True, alpha=0.3)

    # 5. Análisis de décadas (2000s vs 2010s)
    decade_2000s = releases_temp[(releases_temp['year'] >= 2000) & (releases_temp['year'] <= 2009)]
    decade_2010s = releases_temp[(releases_temp['year'] >= 2010) & (releases_temp['year'] <= 2019)]

    decade_data = [len(decade_2000s), len(decade_2010s)]
    decade_labels = ['2000s', '2010s']
    colors = ['lightblue', 'lightcoral']

    bars = axes[2,0].bar(decade_labels, decade_data, color=colors, alpha=0.7)
    axes[2,0].set_title('Comparación de Estrenos por Década')
    axes[2,0].set_ylabel('Número de Estrenos')
    axes[2,0].grid(True, alpha=0.3)

    # 6. Top 5 países por década
    top_countries_2000s = releases_temp[(releases_temp['year'] >= 2000) & (releases_temp['year'] <= 2009)]['country'].value_counts().head(5)
    top_countries_2010s = releases_temp[(releases_temp['year'] >= 2010) & (releases_temp['year'] <= 2019)]['country'].value_counts().head(5)

    x = np.arange(len(top_countries_2000s))
    width = 0.35

    axes[2,1].bar(x - width/2, top_countries_2000s.values, width, label='2000s', alpha=0.7, color='lightblue')
    axes[2,1].bar(x + width/2, top_countries_2010s.values, width, label='2010s', alpha=0.7, color='lightcoral')
    axes[2,1].set_title('Top 5 Países por Número de Estrenos (Comparación por Década)')
    axes[2,1].set_xlabel('País')
    axes[2,1].set_ylabel('Número de Estrenos')
    axes[2,1].set_xticks(x)
    axes[2,1].set_xticklabels(top_countries_2000s.index, rotation=45, ha='right')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/08_reporting/temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {"temporal_analysis": "completed"}


def create_genre_analysis(genres):
    """Crea análisis de géneros y visualizaciones."""
    # Configuración de gráficos
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10

    # Top 15 géneros más frecuentes
    top_genres = genres['genre'].value_counts().head(15)

    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Géneros Cinematográficos', fontsize=16, fontweight='bold')

    # 1. Gráfico de barras horizontal - Top 15 géneros
    axes[0,0].barh(range(len(top_genres)), top_genres.values, color='lightgreen', alpha=0.7)
    axes[0,0].set_yticks(range(len(top_genres)))
    axes[0,0].set_yticklabels(top_genres.index)
    axes[0,0].set_title('Top 15 Géneros Más Frecuentes')
    axes[0,0].set_xlabel('Número de Películas')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Gráfico de pastel - Top 10 géneros
    top_10_genres = genres['genre'].value_counts().head(10)
    other_count = genres['genre'].value_counts().iloc[10:].sum()
    top_10_with_other = top_10_genres.copy()
    top_10_with_other['Otros'] = other_count

    axes[0,1].pie(top_10_with_other.values, labels=top_10_with_other.index, autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('Distribución de Géneros (Top 10 + Otros)')

    # 3. Boxplot de distribución de géneros por película
    movies_genre_count = genres.groupby('id').size()
    axes[1,0].boxplot([movies_genre_count.values], labels=['Géneros por Película'])
    axes[1,0].set_title('Distribución de Número de Géneros por Película')
    axes[1,0].set_ylabel('Número de Géneros')
    axes[1,0].grid(True, alpha=0.3)

    # 4. Histograma de géneros por película
    axes[1,1].hist(movies_genre_count.values, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1,1].set_title('Distribución de Géneros por Película')
    axes[1,1].set_xlabel('Número de Géneros')
    axes[1,1].set_ylabel('Número de Películas')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/08_reporting/genre_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Total de géneros únicos: {genres['genre'].nunique()}")
    print(f"Total de películas: {genres['id'].nunique():,}")
    print(f"Promedio de géneros por película: {movies_genre_count.mean():.2f}")
    print(f"Mediana de géneros por película: {movies_genre_count.median():.0f}")
    print(f"Género más común: {top_genres.index[0]} ({top_genres.iloc[0]:,} películas)")

    return {"genre_analysis": "completed"}


def create_eda_pipeline() -> Pipeline:
    """Crea el pipeline de EDA."""
    return pipeline([
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
        )
    ])
