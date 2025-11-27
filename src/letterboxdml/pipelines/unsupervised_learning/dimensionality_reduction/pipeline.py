"""
Pipeline de Reducción de Dimensionalidad.

Incluye PCA completo, t-SNE y UMAP.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    perform_pca_analysis,
    perform_tsne,
    perform_umap,
    create_pca_loadings_analysis,
    create_biplot_data,
    compare_dimensionality_reduction
)


def create_dimensionality_reduction_pipeline() -> Pipeline:
    """Crear pipeline de reducción de dimensionalidad completo."""
    
    return pipeline([
        node(
            func=perform_pca_analysis,
            inputs="X_scaled",
            outputs="pca_results",
            name="perform_pca_analysis"
        ),
        node(
            func=perform_tsne,
            inputs="X_scaled",
            outputs="tsne_results",
            name="perform_tsne"
        ),
        node(
            func=perform_umap,
            inputs="X_scaled",
            outputs="umap_results",
            name="perform_umap"
        ),
        node(
            func=create_pca_loadings_analysis,
            inputs=["pca_results", "feature_cols_clustering"],
            outputs="pca_loadings",
            name="create_pca_loadings_analysis"
        ),
        node(
            func=create_biplot_data,
            inputs=["pca_results", "feature_cols_clustering"],
            outputs="pca_biplot_data",
            name="create_biplot_data"
        ),
        node(
            func=compare_dimensionality_reduction,
            inputs=["pca_results", "tsne_results", "umap_results"],
            outputs="dim_reduction_comparison",
            name="compare_dimensionality_reduction"
        )
    ])

