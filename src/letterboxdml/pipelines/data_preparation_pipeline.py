"""
Pipeline para Fase 3: Preparación de Datos.

Este módulo contiene todas las funciones necesarias para la preparación
y limpieza de datos cinematográficos, incluyendo:

1. Limpieza de datos (normalización, parsing de fechas, deduplicación)
2. Creación de variables de características (década, año, etc.)
3. Integración de datasets
4. Filtrado y selección de datos relevantes
5. Visualizaciones del proceso de limpieza
6. Análisis comparativo entre décadas

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
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FUNCIONES DE LIMPIEZA DE DATOS
# =============================================================================

def clean_data(releases, countries, genres):
    """
    Limpia los datos: normaliza países, parsea fechas, elimina duplicados.
    
    Esta función realiza la limpieza inicial de los tres datasets principales:
    - Normaliza texto de países (minúsculas, sin puntuación)
    - Parsea fechas de manera robusta
    - Elimina outliers temporales extremos
    - Remueve duplicados exactos
    
    Args:
        releases (pd.DataFrame): Dataset de estrenos
        countries (pd.DataFrame): Dataset de países
        genres (pd.DataFrame): Dataset de géneros
        
    Returns:
        tuple: Tupla con los tres datasets limpios
    """
    # === SELECCIONAR SOLO COLUMNAS RELEVANTES ===
    df_releases = releases[["id", "date"]].copy()
    df_countries = countries[["id", "country"]].copy()
    df_genres = genres[["id", "genre"]].copy()
    
    def _norm_txt(s):
        """
        Normaliza texto: convierte a minúsculas y quita puntos y comas.
        
        Args:
            s: Texto a normalizar
            
        Returns:
            str: Texto normalizado
        """
        if pd.isna(s):
            return ""
        return str(s).lower().replace(".", "").replace(",", "").strip()
    
    # === NORMALIZAR PAÍS (TEXTO) ===
    # No filtra aún, solo normaliza para comparaciones consistentes
    df_countries["_country_norm"] = df_countries["country"].map(_norm_txt)
    
    # === PARSEO ROBUSTO DE FECHA ===
    # Convertir a datetime con manejo de errores
    df_releases["_date_parsed"] = pd.to_datetime(
        df_releases["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    
    # === ELIMINAR FECHAS NO PARSEABLES ===
    _before = len(df_releases)
    df_releases = df_releases.dropna(subset=["_date_parsed"])
    _after = len(df_releases)
    
    # === REMOVER OUTLIERS TEMPORALES EXTREMOS ===
    # Filtrar años razonables (1900-2025) para eliminar datos erróneos
    df_releases = df_releases[df_releases["_date_parsed"].dt.year.between(1900, 2025)]
    
    # === DEDUPLICACIÓN EXACTA POR CLAVES LÓGICAS ===
    # Eliminar duplicados basados en combinaciones de columnas clave
    df_countries = df_countries.drop_duplicates(subset=["id", "country"])
    df_genres = df_genres.drop_duplicates(subset=["id", "genre"])
    df_releases = df_releases.drop_duplicates(subset=["id", "date"])
    
    # === RESUMEN DE LIMPIEZA ===
    print("Limpieza OK.")
    print(f"Fechas no parseables removidas: {_before - _after}")
    print(f"Shapes ⇒ releases: {df_releases.shape}, countries: {df_countries.shape}, genres: {df_genres.shape}")
    
    return df_releases, df_countries, df_genres


def create_features(releases_clean):
    """
    Crea nuevas variables: primera fecha, año, década.
    
    Esta función crea variables de características temporales para cada película:
    - first_date: Primera fecha de estreno por película
    - first_year: Año de la primera fecha
    - decade: Década (2000s, 2010s, other)
    
    Args:
        releases_clean (pd.DataFrame): Dataset de estrenos limpio
        
    Returns:
        pd.DataFrame: Dataset con características temporales creadas
    """
    # === CALCULAR PRIMERA FECHA DE ESTRENO POR PELÍCULA ===
    # Optimizado para usar menos memoria: seleccionar solo columnas necesarias primero
    releases_subset = releases_clean[["id", "_date_parsed"]].copy()
    
    # Usar agg con 'min' directamente para mejor rendimiento
    min_release = releases_subset.groupby("id", as_index=False).agg({
        "_date_parsed": "min"
    }).rename(columns={"_date_parsed": "first_date"})
    
    # Liberar memoria
    del releases_subset
    
    # === EXTRAER AÑO DE LA PRIMERA FECHA DE ESTRENO ===
    # Asegurar que first_date sea datetime
    min_release["first_date"] = pd.to_datetime(min_release["first_date"], errors='coerce')
    min_release["first_year"] = min_release["first_date"].dt.year
    
    # === CLASIFICAR CADA PELÍCULA EN SU DÉCADA CORRESPONDIENTE ===
    # Usar vectorización más eficiente
    min_release["decade"] = min_release["first_year"].apply(
        lambda x: "2000s" if 2000 <= x <= 2009 else ("2010s" if 2010 <= x <= 2019 else "other")
    )
    
    print("Features creadas: first_date, first_year, decade (2000s/2010s/other)")
    print(f"Shape: {min_release.shape}")
    
    return min_release


def integrate_data(min_release, countries_clean, genres_clean):
    """
    Integra datos de múltiples fuentes y aplica filtros finales.
    
    Esta función integra los tres datasets limpios y aplica filtros específicos:
    - Identifica películas de EE. UU. mediante normalización de países
    - Filtra solo décadas de interés (2000s y 2010s)
    - Integra con géneros (multi-etiqueta)
    - Aplica deduplicación final
    
    Args:
        min_release (pd.DataFrame): Dataset con características temporales
        countries_clean (pd.DataFrame): Dataset de países limpio
        genres_clean (pd.DataFrame): Dataset de géneros limpio
        
    Returns:
        pd.DataFrame: Dataset final integrado con filtros aplicados
    """
    # === IDENTIFICAR EE. UU. TRAS NORMALIZACIÓN ===
    # Aliases comunes para Estados Unidos
    US_ALIASES = {"usa", "us", "u s", "u s a", "united states", "united states of america"}
    countries_clean["is_us"] = countries_clean["_country_norm"].isin(US_ALIASES)
    us_ids = set(countries_clean.loc[countries_clean["is_us"], "id"].unique())
    
    # === VERIFICAR INTEGRIDAD BÁSICA DE LLAVES ===
    # Encontrar intersección de IDs entre todos los datasets
    ids_rel = set(min_release["id"].unique())
    ids_gen = set(genres_clean["id"].unique())
    ids_cty = set(countries_clean["id"].unique())
    ids_all = ids_rel & ids_gen & ids_cty
    
    # === CREAR BASE DE PELÍCULAS: EE. UU. + DÉCADAS DE INTERÉS ===
    # Filtrar solo películas de EE. UU. en las décadas 2000s y 2010s
    base = (min_release[min_release["decade"].isin(["2000s", "2010s"])]
            .loc[lambda d: d["id"].isin(us_ids & ids_all), ["id", "first_date", "decade"]]
            .drop_duplicates())
    
    # === UNIR CON GÉNEROS (MULTI-ETIQUETA) Y DEDUPLICAR ===
    # Crear dataset final con combinaciones película-género
    final_df = (base.merge(genres_clean[["id", "genre"]], on="id", how="inner")
                .drop_duplicates(subset=["id", "genre", "decade"])
                [["id", "decade", "genre"]]
                .reset_index(drop=True))
    
    # === RESUMEN DE INTEGRACIÓN ===
    print("=== Integración completada ===")
    print("Filas (id, decade, genre):", final_df.shape[0])
    print("Décadas:", sorted(final_df["decade"].unique().tolist()))
    print("Géneros distintos:", final_df["genre"].nunique())
    
    return final_df


def format_final_data(final_df):
    """
    Formatea y valida el dataset final.
    
    Esta función aplica el formateo final al dataset integrado:
    - Convierte tipos de datos apropiados
    - Establece categorías ordenadas para década
    - Ordena los datos de manera consistente
    - Valida la integridad del dataset
    
    Args:
        final_df (pd.DataFrame): Dataset final integrado
        
    Returns:
        pd.DataFrame: Dataset final formateado y validado
    """
    # === CONVERTIR TIPOS DE DATOS ===
    final_df["id"] = final_df["id"].astype("int64")
    final_df["decade"] = pd.Categorical(final_df["decade"], categories=["2000s", "2010s"], ordered=True)
    final_df["genre"] = final_df["genre"].astype("category")
    
    # === ORDENAR DATOS DE MANERA CONSISTENTE ===
    final_df = final_df.sort_values(["decade", "genre", "id"]).reset_index(drop=True)
    
    # === VERIFICACIONES DE CALIDAD DE DATOS (QA) ===
    assert final_df["id"].isna().sum() == 0, "Hay id nulos"
    assert final_df["genre"].isna().sum() == 0, "Hay genre nulos"
    assert set(final_df["decade"].unique()) <= {"2000s", "2010s"}, "Décadas fuera de rango"
    dup_ok = final_df.duplicated(subset=["id", "genre", "decade"]).sum() == 0
    print("Duplicados (id,genre,decade):", "OK (0)" if dup_ok else "Hay duplicados")
    
    print("\nDataset final listo para análisis del Top-3 por década.")
    print(final_df.dtypes)
    
    return final_df


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def create_cleaning_visualizations(releases, countries, genres, releases_clean, countries_clean, genres_clean, min_release):
    """
    Crea visualizaciones del proceso de limpieza y transformación.
    
    Esta función genera 6 gráficos que muestran el proceso de limpieza:
    1. Antes y después de la limpieza - Conteo de registros
    2. Distribución temporal antes y después
    3. Top países antes y después de normalización
    4. Distribución de géneros antes y después
    5. Análisis de completitud de datos
    6. Resumen del proceso de limpieza
    
    Args:
        releases (pd.DataFrame): Dataset original de estrenos
        countries (pd.DataFrame): Dataset original de países
        genres (pd.DataFrame): Dataset original de géneros
        releases_clean (pd.DataFrame): Dataset limpio de estrenos
        countries_clean (pd.DataFrame): Dataset limpio de países
        genres_clean (pd.DataFrame): Dataset limpio de géneros
        min_release (pd.DataFrame): Dataset con características temporales
        
    Returns:
        dict: Confirmación de visualizaciones completadas
    """
    # === CONFIGURACIÓN DE GRÁFICOS ===
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10

    # Crear figura con subplots para mostrar el proceso de limpieza
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Proceso de Limpieza y Transformación de Datos', fontsize=16, fontweight='bold')

    # 1. Antes y después de la limpieza - Conteo de registros
    datasets_info = {
        'Original': {
            'releases': len(releases),
            'countries': len(countries),
            'genres': len(genres)
        },
        'Después de Limpieza': {
            'releases': len(releases_clean),
            'countries': len(countries_clean),
            'genres': len(genres_clean)
        }
    }

    # Crear DataFrame para visualización
    comparison_data = []
    for stage, datasets in datasets_info.items():
        for dataset, count in datasets.items():
            comparison_data.append({'Etapa': stage, 'Dataset': dataset, 'Registros': count})

    comparison_df = pd.DataFrame(comparison_data)
    comparison_pivot = comparison_df.pivot(index='Dataset', columns='Etapa', values='Registros')

    # Gráfico de barras comparativo
    x = np.arange(len(comparison_pivot.index))
    width = 0.35

    axes[0,0].bar(x - width/2, comparison_pivot['Original'], width, label='Original', alpha=0.7, color='lightcoral')
    axes[0,0].bar(x + width/2, comparison_pivot['Después de Limpieza'], width, label='Después de Limpieza', alpha=0.7, color='lightgreen')
    axes[0,0].set_title('Conteo de Registros: Antes vs Después de Limpieza')
    axes[0,0].set_ylabel('Número de Registros')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(comparison_pivot.index)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Análisis de fechas - Distribución temporal antes y después del filtrado
    releases_original = releases.copy()
    releases_original['date_parsed'] = pd.to_datetime(releases_original['date'], errors='coerce')
    releases_original = releases_original.dropna(subset=['date_parsed'])
    releases_original['year'] = releases_original['date_parsed'].dt.year

    # Filtrar años razonables para visualización
    releases_original = releases_original[(releases_original['year'] >= 1900) & (releases_original['year'] <= 2025)]

    # Datos después de la limpieza
    releases_clean_temp = releases_clean.copy()
    # Asegurar que _date_parsed sea datetime
    releases_clean_temp['_date_parsed'] = pd.to_datetime(releases_clean_temp['_date_parsed'])
    releases_clean_temp['year'] = releases_clean_temp['_date_parsed'].dt.year

    # Crear histogramas comparativos superpuestos
    axes[0,1].hist(releases_original['year'], bins=50, alpha=0.6, label='Original', color='lightcoral', edgecolor='black')
    axes[0,1].hist(releases_clean_temp['year'], bins=50, alpha=0.6, label='Después de Limpieza', color='lightgreen', edgecolor='black')
    axes[0,1].set_title('Distribución Temporal: Antes vs Después de Limpieza')
    axes[0,1].set_xlabel('Año')
    axes[0,1].set_ylabel('Frecuencia')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Análisis de normalización de países - Variantes de USA
    # Verificar que la columna is_us existe
    if 'is_us' in countries_clean.columns:
        us_variants = countries_clean[countries_clean['is_us']]['country'].value_counts()
        if len(us_variants) > 0:
            axes[1,0].pie(us_variants.values, labels=us_variants.index, autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('Variantes de Estados Unidos Detectadas')
        else:
            axes[1,0].text(0.5, 0.5, 'No se encontraron variantes de USA', ha='center', va='center')
            axes[1,0].set_title('Variantes de Estados Unidos Detectadas')
    else:
        axes[1,0].text(0.5, 0.5, 'Columna is_us no encontrada', ha='center', va='center')
        axes[1,0].set_title('Variantes de Estados Unidos Detectadas')

    # 4. Distribución de décadas después del filtrado temporal
    decade_counts = min_release['decade'].value_counts()
    colors = ['lightblue', 'lightcoral', 'lightgray']
    axes[1,1].bar(decade_counts.index, decade_counts.values, color=colors[:len(decade_counts)], alpha=0.7)
    axes[1,1].set_title('Distribución de Películas por Década (Después del Filtrado)')
    axes[1,1].set_ylabel('Número de Películas')
    axes[1,1].grid(True, alpha=0.3)

    # 5. Análisis de duplicados removidos
    duplicates_info = {
        'releases': len(releases) - len(releases.drop_duplicates(subset=['id', 'date'])),
        'countries': len(countries) - len(countries.drop_duplicates(subset=['id', 'country'])),
        'genres': len(genres) - len(genres.drop_duplicates(subset=['id', 'genre']))
    }

    duplicates_df = pd.DataFrame(list(duplicates_info.items()), columns=['Dataset', 'Duplicados_Removidos'])
    axes[2,0].bar(duplicates_df['Dataset'], duplicates_df['Duplicados_Removidos'], color='orange', alpha=0.7)
    axes[2,0].set_title('Duplicados Removidos por Dataset')
    axes[2,0].set_ylabel('Número de Duplicados')
    axes[2,0].grid(True, alpha=0.3)

    # 6. Resumen de la integración final
    us_movies_count = len(countries_clean[countries_clean['is_us']]['id'].unique()) if 'is_us' in countries_clean.columns else 0
    decade_movies_count = len(min_release[min_release['decade'].isin(['2000s', '2010s'])]) if 'decade' in min_release.columns else 0
    unique_genres_count = genres_clean['genre'].nunique() if 'genre' in genres_clean.columns else 0
    
    integration_stats = {
        'Películas EE.UU.': us_movies_count,
        'Películas en 2000s/2010s': decade_movies_count,
        'Géneros Únicos': unique_genres_count
    }

    stats_df = pd.DataFrame(list(integration_stats.items()), columns=['Métrica', 'Valor'])
    axes[2,1].barh(stats_df['Métrica'], stats_df['Valor'], color='lightblue', alpha=0.7)
    axes[2,1].set_title('Estadísticas de Integración Final')
    axes[2,1].set_xlabel('Valor')
    axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/08_reporting/cleaning_process.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {"cleaning_visualizations": "completed"}


def create_comparative_analysis(final_df):
    """Crea análisis comparativo detallado entre décadas 2000s y 2010s."""
    # Configuración de gráficos
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (18, 12)
    plt.rcParams['font.size'] = 10

    # Preparar datos para análisis comparativo
    decade_2000s = final_df[final_df['decade'] == '2000s']
    decade_2010s = final_df[final_df['decade'] == '2010s']

    # Crear figura con subplots para análisis comparativo
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Análisis Comparativo: Década 2000s vs 2010s', fontsize=16, fontweight='bold')

    # 1. Conteo de películas por década
    decade_counts = final_df['decade'].value_counts()
    colors = ['lightblue', 'lightcoral']
    bars = axes[0,0].bar(decade_counts.index, decade_counts.values, color=colors, alpha=0.7)
    axes[0,0].set_title('Número de Películas por Década')
    axes[0,0].set_ylabel('Número de Películas')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Top 10 géneros por década - Comparación lado a lado
    top_genres_2000s = decade_2000s['genre'].value_counts().head(10)
    top_genres_2010s = decade_2010s['genre'].value_counts().head(10)

    # Crear DataFrame para comparación
    comparison_genres = pd.DataFrame({
        '2000s': top_genres_2000s,
        '2010s': top_genres_2010s
    }).fillna(0)

    # Gráfico de barras horizontal lado a lado
    x = np.arange(len(comparison_genres.index))
    width = 0.35

    axes[0,1].barh(x - width/2, comparison_genres['2000s'], width, label='2000s', alpha=0.7, color='lightblue')
    axes[0,1].barh(x + width/2, comparison_genres['2010s'], width, label='2010s', alpha=0.7, color='lightcoral')
    axes[0,1].set_title('Top 10 Géneros por Década')
    axes[0,1].set_xlabel('Número de Películas')
    axes[0,1].set_yticks(x)
    axes[0,1].set_yticklabels(comparison_genres.index)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Distribución de géneros - Gráfico de pastel comparativo
    top5_2000s = decade_2000s['genre'].value_counts().head(5)
    top5_2010s = decade_2010s['genre'].value_counts().head(5)

    # Pastel para 2000s
    wedges1, texts1, autotexts1 = axes[0,2].pie(top5_2000s.values, labels=top5_2000s.index, autopct='%1.1f%%', 
                                       startangle=90, colors=plt.cm.Blues(np.linspace(0.3, 0.8, len(top5_2000s))))
    axes[0,2].set_title('Top 5 Géneros - Década 2000s')

    # 4. Cambio en popularidad de géneros (2000s vs 2010s)
    all_genres = set(decade_2000s['genre'].unique()) | set(decade_2010s['genre'].unique())
    genre_changes = []

    for genre in all_genres:
        count_2000s = len(decade_2000s[decade_2000s['genre'] == genre])
        count_2010s = len(decade_2010s[decade_2010s['genre'] == genre])
        
        if count_2000s > 0:
            change_pct = ((count_2010s - count_2000s) / count_2000s) * 100
        else:
            change_pct = 100 if count_2010s > 0 else 0
        
        genre_changes.append({
            'genre': genre,
            'count_2000s': count_2000s,
            'count_2010s': count_2010s,
            'change_pct': change_pct
        })

    changes_df = pd.DataFrame(genre_changes)
    changes_df = changes_df.sort_values('change_pct', ascending=False)

    # Top 10 géneros con mayor cambio (crecimiento o declive)
    top_changes = changes_df.head(10)
    colors = ['green' if x > 0 else 'red' for x in top_changes['change_pct']]

    bars = axes[1,0].barh(range(len(top_changes)), top_changes['change_pct'], color=colors, alpha=0.7)
    axes[1,0].set_title('Top 10 Géneros con Mayor Cambio (2000s → 2010s)')
    axes[1,0].set_xlabel('Cambio Porcentual (%)')
    axes[1,0].set_yticks(range(len(top_changes)))
    axes[1,0].set_yticklabels(top_changes['genre'])
    axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].grid(True, alpha=0.3)

    # 5. Distribución de géneros por película por década
    movies_2000s = decade_2000s.groupby('id').size()
    movies_2010s = decade_2010s.groupby('id').size()

    # Crear histogramas comparativos
    axes[1,1].hist([movies_2000s.values, movies_2010s.values], bins=15, alpha=0.6, 
                   label=['2000s', '2010s'], color=['lightblue', 'lightcoral'], edgecolor='black')
    axes[1,1].set_title('Distribución de Géneros por Película')
    axes[1,1].set_xlabel('Número de Géneros por Película')
    axes[1,1].set_ylabel('Número de Películas')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    # 6. Pastel para 2010s
    wedges2, texts2, autotexts2 = axes[1,2].pie(top5_2010s.values, labels=top5_2010s.index, autopct='%1.1f%%', 
                                       startangle=90, colors=plt.cm.Reds(np.linspace(0.3, 0.8, len(top5_2010s))))
    axes[1,2].set_title('Top 5 Géneros - Década 2010s')

    plt.tight_layout()
    plt.savefig('data/08_reporting/comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Estadísticas comparativas detalladas
    print("=== ANÁLISIS COMPARATIVO DETALLADO ===")
    print(f"\nPELÍCULAS:")
    print(f"2000s: {len(decade_2000s['id'].unique()):,} películas")
    print(f"2010s: {len(decade_2010s['id'].unique()):,} películas")
    print(f"Crecimiento: {((len(decade_2010s['id'].unique()) - len(decade_2000s['id'].unique())) / len(decade_2000s['id'].unique()) * 100):.1f}%")

    print(f"\nGÉNEROS:")
    print(f"Géneros únicos en 2000s: {decade_2000s['genre'].nunique()}")
    print(f"Géneros únicos en 2010s: {decade_2010s['genre'].nunique()}")

    print(f"\nGÉNEROS MÁS POPULARES:")
    print("2000s:", list(top_genres_2000s.head(3).index))
    print("2010s:", list(top_genres_2010s.head(3).index))

    print(f"\nGÉNEROS CON MAYOR CRECIMIENTO:")
    top_growth = changes_df.head(3)
    for _, row in top_growth.iterrows():
        print(f"{row['genre']}: {row['change_pct']:.1f}% ({row['count_2000s']} → {row['count_2010s']})")

    print(f"\nGÉNEROS CON MAYOR DECLIVE:")
    top_decline = changes_df.tail(3)
    for _, row in top_decline.iterrows():
        print(f"{row['genre']}: {row['change_pct']:.1f}% ({row['count_2000s']} → {row['count_2010s']})")

    print(f"\nGÉNEROS POR PELÍCULA (PROMEDIO):")
    print(f"2000s: {movies_2000s.mean():.2f} géneros por película")
    print(f"2010s: {movies_2010s.mean():.2f} géneros por película")

    return {"comparative_analysis": "completed"}


def save_final_parquet(final_df):
    """Guarda el dataset final en formato Parquet."""
    # Guardar versión limpia en Parquet
    final_df.to_parquet("data/03_primary/final_df_us_2000s_2010s.parquet", index=False)
    print("Guardado: final_df_us_2000s_2010s.parquet")
    
    return {"parquet_saved": "completed"}


# =============================================================================
# FUNCIÓN PRINCIPAL DEL PIPELINE
# =============================================================================

def create_data_preparation_pipeline() -> Pipeline:
    """
    Crea el pipeline de preparación de datos.
    
    Este pipeline incluye todas las etapas de preparación de datos:
    
    LIMPIEZA DE DATOS:
    - Normalización de países
    - Parsing de fechas
    - Eliminación de duplicados
    - Filtrado de outliers temporales
    
    CREACIÓN DE CARACTERÍSTICAS:
    - Primera fecha de estreno por película
    - Año y década
    - Clasificación temporal
    
    INTEGRACIÓN DE DATOS:
    - Identificación de películas de EE. UU.
    - Filtrado por décadas de interés
    - Integración con géneros
    - Deduplicación final
    
    FORMATEO Y VALIDACIÓN:
    - Conversión de tipos de datos
    - Establecimiento de categorías
    - Ordenamiento consistente
    - Validación de integridad
    
    VISUALIZACIONES:
    - Proceso de limpieza
    - Análisis comparativo
    - Guardado en Parquet
    
    Returns:
        Pipeline: Pipeline de Kedro con todos los nodos de preparación
    """
    return pipeline([
        # === LIMPIEZA DE DATOS ===
        node(
            func=clean_data,
            inputs=["releases", "countries", "genres"],
            outputs=["releases_clean", "countries_clean", "genres_clean"],
            name="clean_data"
        ),
        
        # === CREACIÓN DE CARACTERÍSTICAS ===
        node(
            func=create_features,
            inputs=["releases_clean"],
            outputs="min_release",
            name="create_features"
        ),
        
        # === INTEGRACIÓN DE DATOS ===
        node(
            func=integrate_data,
            inputs=["min_release", "countries_clean", "genres_clean"],
            outputs="integrated_data",
            name="integrate_data"
        ),
        
        # === FORMATEO Y VALIDACIÓN ===
        node(
            func=format_final_data,
            inputs=["integrated_data"],
            outputs="final_df",
            name="format_final_data"
        ),
        
        # === VISUALIZACIONES ===
        node(
            func=create_cleaning_visualizations,
            inputs=["releases", "countries", "genres", "releases_clean", "countries_clean", "genres_clean", "min_release"],
            outputs="cleaning_visualizations",
            name="create_cleaning_visualizations"
        ),
        node(
            func=create_comparative_analysis,
            inputs=["final_df"],
            outputs="comparative_analysis",
            name="create_comparative_analysis"
        ),
        
        # === GUARDADO FINAL ===
        node(
            func=save_final_parquet,
            inputs=["final_df"],
            outputs="parquet_saved",
            name="save_final_parquet"
        )
    ])
