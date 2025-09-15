"""Pipeline para Fase 2: Análisis Exploratorio de Datos (EDA)."""
from kedro.pipeline import Pipeline, node, pipeline
import pandas as pd
import numpy as np
from IPython.display import display


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
        )
    ])
