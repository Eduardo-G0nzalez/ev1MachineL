"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.eda_pipeline import create_eda_pipeline
from .pipelines.data_preparation_pipeline import create_data_preparation_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Crear pipelines específicos
    eda_pipeline = create_eda_pipeline()
    data_prep_pipeline = create_data_preparation_pipeline()
    
    # Pipeline por defecto que incluye todos los pipelines
    default_pipeline = eda_pipeline + data_prep_pipeline
    
    return {
        "eda_pipeline": eda_pipeline,
        "data_preparation_pipeline": data_prep_pipeline,
        "__default__": default_pipeline
    }
