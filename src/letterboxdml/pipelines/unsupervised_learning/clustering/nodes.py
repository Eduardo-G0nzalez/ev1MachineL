"""
Nodos para Clustering.

Implementa K-Means, DBSCAN y Clustering Jerárquico.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    silhouette_samples
)
from scipy.cluster.hierarchy import linkage, dendrogram
import json
import warnings
warnings.filterwarnings('ignore')


def prepare_clustering_data(regression_dataset: pd.DataFrame) -> tuple:
    """
    Preparar datos para clustering.
    
    Args:
        regression_dataset: Dataset de regresión con features numéricas
        
    Returns:
        Tuple con (X_scaled, scaler, feature_cols, movie_ids)
    """
    # Seleccionar características numéricas
    numeric_cols = ['minute', 'rating', 'date']
    feature_cols = [col for col in numeric_cols if col in regression_dataset.columns]
    
    # Agregar géneros si existen
    genre_cols = [col for col in regression_dataset.columns if col.startswith('genre_')]
    feature_cols.extend(genre_cols)
    
    # Filtrar datos completos
    df_clean = regression_dataset[feature_cols].dropna()
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols])
    
    # Convertir índices a array numpy para compatibilidad
    movie_indices = df_clean.index.values
    
    return X_scaled, scaler, feature_cols, movie_indices


def find_optimal_k(X_scaled: np.ndarray, k_range: list = None) -> dict:
    """
    Encontrar k óptimo usando Elbow y Silhouette.
    
    Args:
        X_scaled: Datos normalizados
        k_range: Rango de k a probar
        
    Returns:
        Dict con k óptimo y métricas
    """
    if k_range is None:
        k_range = list(range(2, 11))
    
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
    
    # K óptimo según Silhouette
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    # K óptimo según Elbow (segunda derivada)
    if len(inertias) >= 3:
        rate_of_change = np.diff(inertias)
        second_derivative = np.diff(rate_of_change)
        elbow_idx = np.argmax(second_derivative) + 1
        optimal_k_elbow = k_range[min(elbow_idx, len(k_range) - 1)]
    else:
        optimal_k_elbow = optimal_k_silhouette
    
    return {
        'optimal_k_silhouette': optimal_k_silhouette,
        'optimal_k_elbow': optimal_k_elbow,
        'recommended_k': optimal_k_silhouette,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'k_range': k_range
    }


def train_kmeans(X_scaled: np.ndarray, optimal_k_results: dict) -> dict:
    """
    Entrenar modelo K-Means.
    
    Args:
        X_scaled: Datos normalizados
        k_optimal: Número de clusters
        
    Returns:
        Dict con modelo y métricas
    """
    k_optimal = optimal_k_results['recommended_k']
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    
    # Métricas
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    inertia = kmeans.inertia_
    
    return {
        'model': kmeans,
        'labels': labels,
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'inertia': inertia,
        'n_clusters': k_optimal
    }


def train_dbscan(X_scaled: np.ndarray, eps: float = None, min_samples: int = 5) -> dict:
    """
    Entrenar modelo DBSCAN.
    
    Args:
        X_scaled: Datos normalizados
        eps: Radio de búsqueda (si None, busca automáticamente)
        min_samples: Mínimo de muestras por cluster
        
    Returns:
        Dict con modelo y métricas
    """
    if eps is None:
        # Buscar mejor eps
        eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        best_eps = eps_values[0]
        best_silhouette = -1
        
        for eps_test in eps_values:
            dbscan_test = DBSCAN(eps=eps_test, min_samples=min_samples)
            labels_test = dbscan_test.fit_predict(X_scaled)
            n_clusters = len(set(labels_test)) - (1 if -1 in labels_test else 0)
            
            if n_clusters > 1:
                valid_mask = labels_test != -1
                if valid_mask.sum() > 1:
                    silhouette_test = silhouette_score(X_scaled[valid_mask], labels_test[valid_mask])
                    if silhouette_test > best_silhouette:
                        best_silhouette = silhouette_test
                        best_eps = eps_test
        eps = best_eps
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Métricas (solo si hay clusters válidos)
    if n_clusters > 1:
        valid_mask = labels != -1
        if valid_mask.sum() > 1:
            silhouette = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
            davies_bouldin = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
            calinski_harabasz = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
        else:
            silhouette = -1
            davies_bouldin = float('inf')
            calinski_harabasz = 0
    else:
        silhouette = -1
        davies_bouldin = float('inf')
        calinski_harabasz = 0
    
    return {
        'model': dbscan,
        'labels': labels,
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'eps': eps,
        'min_samples': min_samples
    }


def train_hierarchical(X_scaled: np.ndarray, optimal_k_results: dict, linkage_method: str = 'ward') -> dict:
    """
    Entrenar modelo de Clustering Jerárquico.
    
    Args:
        X_scaled: Datos normalizados
        n_clusters: Número de clusters
        linkage_method: Método de linkage ('ward', 'complete', 'average')
        
    Returns:
        Dict con modelo y métricas
    """
    n_clusters = optimal_k_results['recommended_k']
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage=linkage_method
    )
    labels = hierarchical.fit_predict(X_scaled)
    
    # Métricas
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    
    return {
        'model': hierarchical,
        'labels': labels,
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'n_clusters': n_clusters,
        'linkage_method': linkage_method
    }


def evaluate_clustering_models(
    kmeans_results: dict,
    dbscan_results: dict,
    hierarchical_results: dict
) -> pd.DataFrame:
    """
    Evaluar y comparar modelos de clustering.
    
    Args:
        kmeans_results: Resultados de K-Means
        dbscan_results: Resultados de DBSCAN
        hierarchical_results: Resultados de Hierarchical
        
    Returns:
        DataFrame con comparación de métricas
    """
    comparison = {
        'Modelo': ['K-Means', 'DBSCAN', 'Jerárquico'],
        'N_Clusters': [
            kmeans_results['n_clusters'],
            dbscan_results['n_clusters'],
            hierarchical_results['n_clusters']
        ],
        'Silhouette_Score': [
            kmeans_results['silhouette_score'],
            dbscan_results['silhouette_score'] if dbscan_results['n_clusters'] > 1 else np.nan,
            hierarchical_results['silhouette_score']
        ],
        'Davies_Bouldin': [
            kmeans_results['davies_bouldin_index'],
            dbscan_results['davies_bouldin_index'] if dbscan_results['n_clusters'] > 1 else np.nan,
            hierarchical_results['davies_bouldin_index']
        ],
        'Calinski_Harabasz': [
            kmeans_results['calinski_harabasz_score'],
            dbscan_results['calinski_harabasz_score'] if dbscan_results['n_clusters'] > 1 else np.nan,
            hierarchical_results['calinski_harabasz_score']
        ]
    }
    
    return pd.DataFrame(comparison)


def save_clustering_metrics(
    clustering_comparison: pd.DataFrame
) -> dict:
    """
    Guardar métricas de clustering en formato JSON.
    
    Args:
        clustering_comparison: DataFrame con comparación de modelos
        
    Returns:
        Dict con métricas para DVC
    """
    metrics_dict = {}
    
    for _, row in clustering_comparison.iterrows():
        model_name = row['Modelo']
        metrics_dict[model_name] = {
            'n_clusters': int(row['N_Clusters']),
            'silhouette_score': float(row['Silhouette_Score']) if not pd.isna(row['Silhouette_Score']) else None,
            'davies_bouldin_index': float(row['Davies_Bouldin']) if not pd.isna(row['Davies_Bouldin']) else None,
            'calinski_harabasz_score': float(row['Calinski_Harabasz']) if not pd.isna(row['Calinski_Harabasz']) else None
        }
    
    return metrics_dict

