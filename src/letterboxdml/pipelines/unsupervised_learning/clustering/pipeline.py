"""
Pipeline de Clustering.

Incluye K-Means, DBSCAN y Clustering Jerárquico.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_clustering_data,
    find_optimal_k,
    train_kmeans,
    train_dbscan,
    train_hierarchical,
    evaluate_clustering_models
)


def create_clustering_pipeline() -> Pipeline:
    """Crear pipeline de clustering completo."""
    
    return pipeline([
        node(
            func=prepare_clustering_data,
            inputs="regression_dataset",
            outputs=["X_scaled", "scaler_clustering", "feature_cols_clustering", "movie_indices"],
            name="prepare_clustering_data"
        ),
        node(
            func=find_optimal_k,
            inputs="X_scaled",
            outputs="optimal_k_results",
            name="find_optimal_k"
        ),
        node(
            func=train_kmeans,
            inputs=["X_scaled", "optimal_k_results"],
            outputs="kmeans_results",
            name="train_kmeans"
        ),
        node(
            func=train_dbscan,
            inputs="X_scaled",
            outputs="dbscan_results",
            name="train_dbscan"
        ),
        node(
            func=train_hierarchical,
            inputs=["X_scaled", "optimal_k_results"],
            outputs="hierarchical_results",
            name="train_hierarchical"
        ),
        node(
            func=evaluate_clustering_models,
            inputs=["kmeans_results", "dbscan_results", "hierarchical_results"],
            outputs="clustering_comparison",
            name="evaluate_clustering_models"
        )
    ])

