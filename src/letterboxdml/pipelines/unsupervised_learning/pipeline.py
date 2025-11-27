"""
Pipeline maestro de Aprendizaje No Supervisado.

Combina clustering y reducción de dimensionalidad.
"""

from kedro.pipeline import Pipeline, pipeline
from .clustering.pipeline import create_clustering_pipeline
from .dimensionality_reduction.pipeline import create_dimensionality_reduction_pipeline
from .integration.nodes import add_cluster_features


def create_unsupervised_learning_pipeline() -> Pipeline:
    """
    Crear pipeline completo de aprendizaje no supervisado.
    
    Incluye:
    1. Clustering (K-Means, DBSCAN, Hierarchical)
    2. Reducción de Dimensionalidad (PCA, t-SNE, UMAP)
    3. Integración con modelos supervisados
    """
    clustering_pipe = create_clustering_pipeline()
    dim_reduction_pipe = create_dimensionality_reduction_pipeline()
    
    # Pipeline combinado
    unsupervised_pipeline = clustering_pipe + dim_reduction_pipe
    
    return unsupervised_pipeline

