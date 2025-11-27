"""
Nodos para integrar clustering con modelos supervisados.

Usa clusters como features adicionales para mejorar modelos supervisados.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def add_cluster_features(
    classification_dataset: pd.DataFrame,
    regression_dataset: pd.DataFrame,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
    movie_indices: np.ndarray
) -> tuple:
    """
    Agregar features de clustering a datasets de clasificación y regresión.
    
    Args:
        classification_dataset: Dataset de clasificación
        regression_dataset: Dataset de regresión
        kmeans_labels: Etiquetas de K-Means
        dbscan_labels: Etiquetas de DBSCAN
        hierarchical_labels: Etiquetas de Hierarchical
        movie_indices: Índices de películas usadas en clustering
        
    Returns:
        Tupla con datasets actualizados
    """
    # Crear DataFrame con clusters
    clusters_df = pd.DataFrame({
        'id': movie_indices,
        'cluster_kmeans': kmeans_labels,
        'cluster_dbscan': dbscan_labels,
        'cluster_hierarchical': hierarchical_labels
    })
    
    # Agregar a classification dataset
    if 'id' in classification_dataset.columns:
        classification_with_clusters = classification_dataset.merge(
            clusters_df[['id', 'cluster_kmeans', 'cluster_dbscan', 'cluster_hierarchical']],
            on='id',
            how='left'
        )
        # Codificar como one-hot si es necesario
        classification_with_clusters['cluster_kmeans_encoded'] = classification_with_clusters['cluster_kmeans'].fillna(-1)
        classification_with_clusters['cluster_dbscan_encoded'] = classification_with_clusters['cluster_dbscan'].fillna(-1)
        classification_with_clusters['cluster_hierarchical_encoded'] = classification_with_clusters['cluster_hierarchical'].fillna(-1)
    else:
        classification_with_clusters = classification_dataset.copy()
    
    # Agregar a regression dataset
    if 'id' in regression_dataset.columns:
        regression_with_clusters = regression_dataset.merge(
            clusters_df[['id', 'cluster_kmeans', 'cluster_dbscan', 'cluster_hierarchical']],
            on='id',
            how='left'
        )
        regression_with_clusters['cluster_kmeans_encoded'] = regression_with_clusters['cluster_kmeans'].fillna(-1)
        regression_with_clusters['cluster_dbscan_encoded'] = regression_with_clusters['cluster_dbscan'].fillna(-1)
        regression_with_clusters['cluster_hierarchical_encoded'] = regression_with_clusters['cluster_hierarchical'].fillna(-1)
    else:
        regression_with_clusters = regression_dataset.copy()
    
    return classification_with_clusters, regression_with_clusters


def compare_models_with_without_clusters(
    metrics_without_clusters: dict,
    metrics_with_clusters: dict
) -> pd.DataFrame:
    """
    Comparar métricas de modelos con y sin clusters.
    
    Args:
        metrics_without_clusters: Métricas sin clusters
        metrics_with_clusters: Métricas con clusters
        
    Returns:
        DataFrame con comparación
    """
    comparison = {
        'Modelo': [],
        'Métrica': [],
        'Sin_Clusters': [],
        'Con_Clusters': [],
        'Mejora': []
    }
    
    # Comparar clasificación
    if 'classification' in metrics_without_clusters and 'classification' in metrics_with_clusters:
        for metric_name in ['accuracy', 'f1_score', 'precision', 'recall']:
            if metric_name in metrics_without_clusters['classification']:
                without = metrics_without_clusters['classification'][metric_name]
                with_clusters = metrics_with_clusters['classification'][metric_name]
                improvement = with_clusters - without
                
                comparison['Modelo'].append('Clasificación')
                comparison['Métrica'].append(metric_name)
                comparison['Sin_Clusters'].append(without)
                comparison['Con_Clusters'].append(with_clusters)
                comparison['Mejora'].append(improvement)
    
    # Comparar regresión
    if 'regression' in metrics_without_clusters and 'regression' in metrics_with_clusters:
        for metric_name in ['r2_score', 'rmse', 'mae']:
            if metric_name in metrics_without_clusters['regression']:
                without = metrics_without_clusters['regression'][metric_name]
                with_clusters = metrics_with_clusters['regression'][metric_name]
                # Para RMSE y MAE, menor es mejor, así que mejora es negativa
                if metric_name in ['rmse', 'mae']:
                    improvement = without - with_clusters
                else:
                    improvement = with_clusters - without
                
                comparison['Modelo'].append('Regresión')
                comparison['Métrica'].append(metric_name)
                comparison['Sin_Clusters'].append(without)
                comparison['Con_Clusters'].append(with_clusters)
                comparison['Mejora'].append(improvement)
    
    return pd.DataFrame(comparison)

