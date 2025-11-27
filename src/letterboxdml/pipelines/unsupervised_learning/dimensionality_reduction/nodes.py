"""
Nodos para Reducción de Dimensionalidad.

Implementa PCA completo, t-SNE y UMAP.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Importar UMAP de forma opcional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP no está instalado. Instalar con: pip install umap-learn")


def perform_pca_analysis(X_scaled: np.ndarray, n_components: int = None) -> dict:
    """
    Realizar análisis completo de PCA.
    
    Args:
        X_scaled: Datos normalizados
        n_components: Número de componentes (None = todos)
        
    Returns:
        Dict con modelo PCA, varianza explicada, loadings, etc.
    """
    if n_components is None:
        n_components = min(X_scaled.shape[1], X_scaled.shape[0])
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Varianza explicada
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Loadings (componentes principales)
    components = pca.components_
    
    # Determinar número óptimo de componentes (95% varianza)
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    return {
        'model': pca,
        'transformed_data': X_pca,
        'explained_variance_ratio': explained_variance,
        'cumulative_variance_ratio': cumulative_variance,
        'components': components,
        'n_components_95': n_components_95,
        'n_components': n_components,
        'feature_names': [f'PC{i+1}' for i in range(n_components)]
    }


def perform_tsne(X_scaled: np.ndarray, n_components: int = 2, perplexity: float = 30.0, 
                  max_iter: int = 1000, random_state: int = 42) -> dict:
    """
    Realizar t-SNE para visualización.
    
    Args:
        X_scaled: Datos normalizados
        n_components: Dimensiones de salida (2 o 3)
        perplexity: Parámetro de perplexity
        max_iter: Número máximo de iteraciones
        random_state: Semilla aleatoria
        
    Returns:
        Dict con modelo y datos transformados
    """
    # Limitar tamaño para t-SNE (es computacionalmente costoso)
    max_samples = 10000
    if X_scaled.shape[0] > max_samples:
        indices = np.random.choice(X_scaled.shape[0], max_samples, replace=False)
        X_sample = X_scaled[indices]
        sample_indices = indices
    else:
        X_sample = X_scaled
        sample_indices = np.arange(X_scaled.shape[0])
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )
    
    X_tsne = tsne.fit_transform(X_sample)
    
    return {
        'model': tsne,
        'transformed_data': X_tsne,
        'sample_indices': sample_indices,
        'n_components': n_components,
        'perplexity': perplexity,
        'kl_divergence': tsne.kl_divergence_
    }


def perform_umap(X_scaled: np.ndarray, n_components: int = 2, n_neighbors: int = 15,
                  min_dist: float = 0.1, random_state: int = 42) -> dict:
    """
    Realizar UMAP para reducción de dimensionalidad.
    
    Args:
        X_scaled: Datos normalizados
        n_components: Dimensiones de salida
        n_neighbors: Número de vecinos
        min_dist: Distancia mínima
        random_state: Semilla aleatoria
        
    Returns:
        Dict con modelo y datos transformados
    """
    if not UMAP_AVAILABLE:
        # Si UMAP no está disponible, usar PCA como alternativa
        warnings.warn("UMAP no disponible, usando PCA como alternativa")
        pca = PCA(n_components=n_components, random_state=random_state)
        X_umap = pca.fit_transform(X_scaled)
        return {
            'model': pca,
            'transformed_data': X_umap,
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'fallback_to_pca': True
        }
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    X_umap = reducer.fit_transform(X_scaled)
    
    return {
        'model': reducer,
        'transformed_data': X_umap,
        'n_components': n_components,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'fallback_to_pca': False
    }


def create_pca_loadings_analysis(pca_results: dict, feature_names: list) -> pd.DataFrame:
    """
    Crear análisis de loadings de PCA.
    
    Args:
        pca_results: Resultados de PCA
        feature_names: Nombres de las features originales
        
    Returns:
        DataFrame con loadings por componente
    """
    components = pca_results['components']
    n_components = min(10, components.shape[0])  # Primeros 10 componentes
    
    loadings_data = []
    for i in range(n_components):
        for j, feature_name in enumerate(feature_names[:components.shape[1]]):
            loadings_data.append({
                'Component': f'PC{i+1}',
                'Feature': feature_name,
                'Loading': components[i, j],
                'Variance_Explained': pca_results['explained_variance_ratio'][i]
            })
    
    return pd.DataFrame(loadings_data)


def create_biplot_data(pca_results: dict, feature_names: list, n_features: int = None) -> dict:
    """
    Preparar datos para biplot de PCA.
    
    Args:
        pca_results: Resultados de PCA
        feature_names: Nombres de las features
        n_features: Número de features a mostrar (None = todas)
        
    Returns:
        Dict con datos para biplot
    """
    components = pca_results['components']
    explained_variance = pca_results['explained_variance_ratio']
    
    # Seleccionar features más importantes
    if n_features is None:
        n_features = min(10, len(feature_names))
    
    # Calcular importancia de features (suma de valores absolutos de loadings)
    feature_importance = np.abs(components[:2, :]).sum(axis=0)
    top_indices = np.argsort(feature_importance)[-n_features:]
    
    return {
        'components': components[:2, top_indices],  # PC1 y PC2
        'feature_names': [feature_names[i] for i in top_indices],
        'explained_variance_pc1': explained_variance[0],
        'explained_variance_pc2': explained_variance[1],
        'top_indices': top_indices
    }


def compare_dimensionality_reduction(
    pca_results: dict,
    tsne_results: dict,
    umap_results: dict
) -> pd.DataFrame:
    """
    Comparar técnicas de reducción de dimensionalidad.
    
    Args:
        pca_results: Resultados de PCA
        tsne_results: Resultados de t-SNE
        umap_results: Resultados de UMAP
        
    Returns:
        DataFrame con comparación
    """
    comparison = {
        'Técnica': ['PCA', 't-SNE', 'UMAP'],
        'N_Componentes': [
            pca_results['n_components'],
            tsne_results['n_components'],
            umap_results['n_components']
        ],
        'Varianza_Explicada_PC1': [
            pca_results['explained_variance_ratio'][0] if len(pca_results['explained_variance_ratio']) > 0 else 0,
            np.nan,  # t-SNE no tiene varianza explicada
            np.nan   # UMAP no tiene varianza explicada
        ],
        'Varianza_Explicada_PC2': [
            pca_results['explained_variance_ratio'][1] if len(pca_results['explained_variance_ratio']) > 1 else 0,
            np.nan,
            np.nan
        ],
        'Varianza_Acumulada_2D': [
            pca_results['cumulative_variance_ratio'][1] if len(pca_results['cumulative_variance_ratio']) > 1 else 0,
            np.nan,
            np.nan
        ],
        'Muestras_Procesadas': [
            pca_results['transformed_data'].shape[0],
            tsne_results['transformed_data'].shape[0],
            umap_results['transformed_data'].shape[0]
        ]
    }
    
    return pd.DataFrame(comparison)

