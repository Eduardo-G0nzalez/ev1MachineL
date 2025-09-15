"""Pipeline para Fase 3: Preparación de Datos."""
from kedro.pipeline import Pipeline, node, pipeline
import pandas as pd
import numpy as np


def clean_data(releases, countries, genres):
    """Limpia los datos: normaliza países, parsea fechas, elimina duplicados."""
    # Seleccionar solo columnas relevantes
    df_releases = releases[["id", "date"]].copy()
    df_countries = countries[["id", "country"]].copy()
    df_genres = genres[["id", "genre"]].copy()
    
    def _norm_txt(s):
        """Normaliza texto: convierte a minúsculas y quita puntos y comas."""
        if pd.isna(s):
            return ""
        return str(s).lower().replace(".", "").replace(",", "").strip()
    
    # Normalizar país (texto) – no filtra aún
    df_countries["_country_norm"] = df_countries["country"].map(_norm_txt)
    
    # Parseo robusto de fecha (naive)
    df_releases["_date_parsed"] = pd.to_datetime(
        df_releases["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    
    # Drop de fechas no parseables
    _before = len(df_releases)
    df_releases = df_releases.dropna(subset=["_date_parsed"])
    _after = len(df_releases)
    
    # Remueve outliers temporales extremos (muy fuera de rango razonable)
    df_releases = df_releases[df_releases["_date_parsed"].dt.year.between(1900, 2025)]
    
    # Deduplicación exacta por claves lógicas
    df_countries = df_countries.drop_duplicates(subset=["id", "country"])
    df_genres = df_genres.drop_duplicates(subset=["id", "genre"])
    df_releases = df_releases.drop_duplicates(subset=["id", "date"])
    
    print("Limpieza OK.")
    print(f"Fechas no parseables removidas: {_before - _after}")
    print(f"Shapes ⇒ releases: {df_releases.shape}, countries: {df_countries.shape}, genres: {df_genres.shape}")
    
    return df_releases, df_countries, df_genres


def create_features(releases_clean):
    """Crea nuevas variables: primera fecha, año, década."""
    # Calcula la primera fecha de estreno por película (fecha mínima por id)
    min_release = (releases_clean.groupby("id", as_index=False)["_date_parsed"].min()
                   .rename(columns={"_date_parsed": "first_date"}))
    
    # Extrae el año de la primera fecha de estreno
    # Asegurar que first_date sea datetime
    min_release["first_date"] = pd.to_datetime(min_release["first_date"])
    min_release["first_year"] = min_release["first_date"].dt.year
    
    # Clasifica cada película en su década correspondiente
    min_release["decade"] = np.where(min_release["first_year"].between(2000, 2009), "2000s",
                             np.where(min_release["first_year"].between(2010, 2019), "2010s", "other"))
    
    print("Features creadas: first_date, first_year, decade (2000s/2010s/other)")
    
    return min_release


def integrate_data(min_release, countries_clean, genres_clean):
    """Integra datos de múltiples fuentes y aplica filtros finales."""
    # Identificar EE. UU. tras normalización
    US_ALIASES = {"usa", "us", "u s", "u s a", "united states", "united states of america"}
    countries_clean["is_us"] = countries_clean["_country_norm"].isin(US_ALIASES)
    us_ids = set(countries_clean.loc[countries_clean["is_us"], "id"].unique())
    
    # Integridad básica de llaves
    ids_rel = set(min_release["id"].unique())
    ids_gen = set(genres_clean["id"].unique())
    ids_cty = set(countries_clean["id"].unique())
    ids_all = ids_rel & ids_gen & ids_cty
    
    # Base de películas: EE. UU. + décadas de interés (2000s/2010s)
    base = (min_release[min_release["decade"].isin(["2000s", "2010s"])]
            .loc[lambda d: d["id"].isin(us_ids & ids_all), ["id", "first_date", "decade"]]
            .drop_duplicates())
    
    # Unir con géneros (multi-etiqueta) y deduplicar
    final_df = (base.merge(genres_clean[["id", "genre"]], on="id", how="inner")
                .drop_duplicates(subset=["id", "genre", "decade"])
                [["id", "decade", "genre"]]
                .reset_index(drop=True))
    
    print("=== Integración completada ===")
    print("Filas (id, decade, genre):", final_df.shape[0])
    print("Décadas:", sorted(final_df["decade"].unique().tolist()))
    print("Géneros distintos:", final_df["genre"].nunique())
    
    return final_df


def format_final_data(final_df):
    """Formatea y valida el dataset final."""
    # Tipos y orden
    final_df["id"] = final_df["id"].astype("int64")
    final_df["decade"] = pd.Categorical(final_df["decade"], categories=["2000s", "2010s"], ordered=True)
    final_df["genre"] = final_df["genre"].astype("category")
    
    final_df = final_df.sort_values(["decade", "genre", "id"]).reset_index(drop=True)
    
    # QA rápido (verificaciones de calidad de datos)
    assert final_df["id"].isna().sum() == 0, "Hay id nulos"
    assert final_df["genre"].isna().sum() == 0, "Hay genre nulos"
    assert set(final_df["decade"].unique()) <= {"2000s", "2010s"}, "Décadas fuera de rango"
    dup_ok = final_df.duplicated(subset=["id", "genre", "decade"]).sum() == 0
    print("Duplicados (id,genre,decade):", "OK (0)" if dup_ok else "Hay duplicados")
    
    print("\nDataset final listo para análisis del Top-3 por década.")
    print(final_df.dtypes)
    
    return final_df


def create_data_preparation_pipeline() -> Pipeline:
    """Crea el pipeline de preparación de datos."""
    return pipeline([
        node(
            func=clean_data,
            inputs=["releases", "countries", "genres"],
            outputs=["releases_clean", "countries_clean", "genres_clean"],
            name="clean_data"
        ),
        node(
            func=create_features,
            inputs=["releases_clean"],
            outputs="min_release",
            name="create_features"
        ),
        node(
            func=integrate_data,
            inputs=["min_release", "countries_clean", "genres_clean"],
            outputs="integrated_data",
            name="integrate_data"
        ),
        node(
            func=format_final_data,
            inputs=["integrated_data"],
            outputs="final_df",
            name="format_final_data"
        )
    ])
