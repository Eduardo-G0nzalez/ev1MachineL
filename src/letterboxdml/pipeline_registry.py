"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.eda_pipeline import create_eda_pipeline
from .pipelines.data_preparation_pipeline import create_data_preparation_pipeline
from .pipelines.ml_modeling_pipeline import (
    create_ml_modeling_pipeline,
    create_classification_pipeline,
    create_regression_pipeline
)
from .pipelines.unsupervised_learning.pipeline import create_unsupervised_learning_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Crear pipelines específicos
    eda_pipeline = create_eda_pipeline()
    data_prep_pipeline = create_data_preparation_pipeline()
    
    # Pipelines de ML
    classification_pipeline = create_classification_pipeline()
    regression_pipeline = create_regression_pipeline()
    ml_modeling_pipeline = create_ml_modeling_pipeline()
    
    # Pipeline de Unsupervised Learning
    unsupervised_pipeline = create_unsupervised_learning_pipeline()
    
    # Pipeline por defecto que incluye todos los pipelines
    default_pipeline = eda_pipeline + data_prep_pipeline + ml_modeling_pipeline + unsupervised_pipeline
    
    return {
        "eda_pipeline": eda_pipeline,
        "data_preparation_pipeline": data_prep_pipeline,
        "classification_pipeline": classification_pipeline,
        "regression_pipeline": regression_pipeline,
        "ml_modeling_pipeline": ml_modeling_pipeline,
        "unsupervised_learning_pipeline": unsupervised_pipeline,
        "__default__": default_pipeline
    }
