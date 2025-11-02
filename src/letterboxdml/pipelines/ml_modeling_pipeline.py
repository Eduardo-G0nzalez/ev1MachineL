"""
Pipeline para Fase 4: Modelado de Machine Learning (Clasificación y Regresión).

Este módulo contiene pipelines completos para:
1. Clasificación: Predecir nivel de éxito comercial (Alto/Medio/Bajo)
2. Regresión: Predecir rating de películas (0-5)

Autores: Mathias Jara & Eduardo Gonzalez
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline import pipeline as pipe
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, classification_report, confusion_matrix
)
import pickle
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FUNCIONES DE PREPARACIÓN DE DATOS
# =============================================================================

def prepare_classification_data(releases, countries, genres, movies):
    """Preparar datos para clasificación basado en rating por edad"""
    
    # Identificar EE.UU.
    us_aliases = {"usa", "us", "u s", "u s a", "united states", "united states of america"}
    countries['country_lower'] = countries['country'].str.lower().str.strip()
    countries['is_us'] = countries['country_lower'].isin(us_aliases)
    us_ids = set(countries[countries['is_us']]['id'].unique())
    
    # Filtrar releases de EE.UU. con rating válido
    us_releases = releases[releases['id'].isin(us_ids)].copy()
    us_releases['date'] = pd.to_datetime(us_releases['date'], errors='coerce')
    us_releases = us_releases.dropna(subset=['date'])
    us_releases['year'] = us_releases['date'].dt.year
    us_releases['decade'] = us_releases['year'].apply(
        lambda x: '2000s' if 2000 <= x <= 2009 else ('2010s' if 2010 <= x <= 2019 else 'other')
    )
    
    # Filtrar solo 2000s y 2010s con rating no nulo
    us_releases = us_releases[
        (us_releases['decade'].isin(['2000s', '2010s'])) & 
        (us_releases['rating'].notna())
    ].copy()
    
    # Agregar información de películas
    movies_filtered = movies[['id', 'minute']].copy()
    us_releases = us_releases.merge(movies_filtered, on='id', how='left')
    
    # Codificar géneros
    genres_filtered = genres[genres['id'].isin(us_releases['id'].unique())]
    genre_dummies = pd.get_dummies(genres_filtered['genre'], prefix='genre')
    genre_agg = genres_filtered[['id']].merge(genre_dummies, left_index=True, right_index=True)
    genre_agg = genre_agg.groupby('id').sum().reset_index()
    
    # Merge final
    df_final = us_releases.merge(genre_agg, on='id', how='left')
    
    # Mapear ratings a niveles de éxito
    def map_rating_success(rating):
        if rating in ['G', 'PG', 'PG-13']:
            return 'Alto'
        elif rating in ['PG-13']:
            return 'Medio'
        else:  # R, NC-17
            return 'Bajo'
    
    df_final['success_level'] = df_final['rating'].apply(map_rating_success)
    
    # Seleccionar features
    genre_cols = [col for col in df_final.columns if col.startswith('genre_')]
    feature_cols = ['minute', 'decade'] + genre_cols
    feature_cols = [col for col in feature_cols if col in df_final.columns]
    
    # Remover valores faltantes
    df_ml = df_final[feature_cols + ['success_level']].dropna()
    
    return df_ml


def prepare_regression_data(movies, genres):
    """Preparar datos para regresión basado en rating de películas"""
    
    # Filtrar películas con rating válido en décadas 2000s y 2010s
    movies_filtered = movies[
        (movies['rating'].notna()) & 
        (movies['date'].between(2000, 2019)) &
        (movies['minute'].notna())
    ].copy()
    
    movies_filtered['decade'] = movies_filtered['date'].apply(
        lambda x: '2000s' if 2000 <= x <= 2009 else ('2010s' if 2010 <= x <= 2019 else 'other')
    )
    movies_filtered = movies_filtered[movies_filtered['decade'].isin(['2000s', '2010s'])]
    
    # Codificar géneros
    genres_filtered = genres[genres['id'].isin(movies_filtered['id'].unique())]
    genre_dummies = pd.get_dummies(genres_filtered['genre'], prefix='genre')
    genre_agg = genres_filtered[['id']].merge(genre_dummies, left_index=True, right_index=True)
    genre_agg = genre_agg.groupby('id').sum().reset_index()
    
    # Merge final
    df_final = movies_filtered.merge(genre_agg, on='id', how='inner')
    
    # Seleccionar features
    genre_cols = [col for col in df_final.columns if col.startswith('genre_')]
    feature_cols = ['minute', 'date'] + genre_cols
    feature_cols = [col for col in feature_cols if col in df_final.columns]
    
    df_ml = df_final[feature_cols + ['rating']].dropna()
    
    return df_ml


# =============================================================================
# FUNCIONES DE ENTRENAMIENTO CON GRIDSEARCH Y CV
# =============================================================================

def train_classification_models(df_classification):
    """Entrenar 5 modelos de clasificación con GridSearchCV y CV"""
    
    # Preparar datos
    genre_cols = [col for col in df_classification.columns if col.startswith('genre_')]
    feature_cols = ['minute', 'decade'] + genre_cols
    feature_cols = [col for col in feature_cols if col in df_classification.columns]
    
    X = df_classification[feature_cols].copy()
    y = df_classification['success_level']
    
    # Codificar columna 'decade' si es string
    if 'decade' in X.columns and X['decade'].dtype == 'object':
        decade_le = LabelEncoder()
        X['decade_encoded'] = decade_le.fit_transform(X['decade'])
        X = X.drop('decade', axis=1)
    
    # Codificar target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configuración de modelos con GridSearch
    models_config = {
        'Logistic_Regression': {
            'model': LogisticRegression(random_state=42),
            'param_grid': {'C': [0.1, 1, 10], 'max_iter': [1000]},
            'cv': 5
        },
        'Decision_Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'param_grid': {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5]},
            'cv': 5
        },
        'Random_Forest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {'n_estimators': [50, 100], 'max_depth': [10, 15]},
            'cv': 5
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'param_grid': {'n_neighbors': [3, 5, 7]},
            'cv': 5
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'param_grid': {'C': [0.1, 1], 'gamma': ['scale', 'auto']},
            'cv': 3
        }
    }
    
    results = {}
    
    print("🚀 Entrenando modelos de clasificación con GridSearchCV...\n")
    
    for name, config in models_config.items():
        print(f"📊 {name}...")
        
        # GridSearchCV
        grid_search = GridSearchCV(
            config['model'],
            config['param_grid'],
            cv=config['cv'],
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # Predicciones
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
        
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'scaler': scaler,
            'label_encoder': le
        }
        
        print(f"   ✅ F1: {f1:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    return results


def train_regression_models(df_regression):
    """Entrenar 5 modelos de regresión con GridSearchCV y CV"""
    
    # Preparar datos
    genre_cols = [col for col in df_regression.columns if col.startswith('genre_')]
    feature_cols = ['minute', 'date'] + genre_cols
    feature_cols = [col for col in feature_cols if col in df_regression.columns]
    
    X = df_regression[feature_cols]
    y = df_regression['rating']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configuración de modelos con GridSearch
    models_config = {
        'Linear_Regression': {
            'model': pd.DataFrame(),  # No grid search para LR
            'param_grid': {},
            'cv': 5
        },
        'Random_Forest': {
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {'n_estimators': [50, 100], 'max_depth': [10, 15]},
            'cv': 5
        },
        'Gradient_Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'param_grid': {'n_estimators': [50, 100], 'max_depth': [3, 5]},
            'cv': 5
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'param_grid': {'n_neighbors': [3, 5, 7]},
            'cv': 5
        },
        'SVR': {
            'model': SVR(),
            'param_grid': {'C': [0.1, 1], 'gamma': ['scale', 'auto']},
            'cv': 3
        }
    }
    
    results = {}
    
    print("🚀 Entrenando modelos de regresión con GridSearchCV...\n")
    
    for name, config in models_config.items():
        print(f"📊 {name}...")
        
        # Para Linear Regression, no usar GridSearch
        if name == 'Linear_Regression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            best_model = model
            best_params = {}
        else:
            grid_search = GridSearchCV(
                config['model'],
                config['param_grid'],
                cv=config['cv'],
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        
        # Predicciones
        y_pred = best_model.predict(X_test_scaled)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            best_model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        results[name] = {
            'model': best_model,
            'best_params': best_params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'explained_var': explained_var,
            'cv_mean': cv_rmse.mean(),
            'cv_std': cv_rmse.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'scaler': scaler
        }
        
        print(f"   ✅ R²: {r2:.4f} | RMSE: {rmse:.4f} | CV RMSE: {cv_rmse.mean():.4f}±{cv_rmse.std():.4f}")
    
    return results


# =============================================================================
# FUNCIONES DE EVALUACIÓN Y REPORTES
# =============================================================================

def evaluate_classification_models(classification_results):
    """Evaluar y generar reportes de clasificación"""
    
    # Crear DataFrame comparativo
    metrics_data = []
    
    for name, result in classification_results.items():
        metrics_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1'],
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Guardar resultados
    print("\n📊 Tabla Comparativa de Clasificación:")
    print(metrics_df.round(4))
    
    return {"classification_metrics": metrics_df}


def evaluate_regression_models(regression_results):
    """Evaluar y generar reportes de regresión"""
    
    # Crear DataFrame comparativo
    metrics_data = []
    
    for name, result in regression_results.items():
        metrics_data.append({
            'Model': name,
            'R2_Score': result['r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'MSE': result['mse'],
            'CV_RMSE_Mean': result['cv_mean'],
            'CV_RMSE_Std': result['cv_std']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Guardar resultados
    print("\n📊 Tabla Comparativa de Regresión:")
    print(metrics_df.round(4))
    
    return {"regression_metrics": metrics_df}


# =============================================================================
# FUNCIONES PRINCIPALES DE PIPELINES
# =============================================================================

def create_classification_pipeline() -> Pipeline:
    """Pipeline de clasificación completo"""
    return pipe([
        node(
            func=prepare_classification_data,
            inputs=["releases", "countries", "genres", "movies"],
            outputs="classification_dataset",
            name="prepare_classification_data"
        ),
        node(
            func=train_classification_models,
            inputs="classification_dataset",
            outputs="classification_results",
            name="train_classification_models"
        ),
        node(
            func=evaluate_classification_models,
            inputs="classification_results",
            outputs="classification_metrics",
            name="evaluate_classification_models"
        )
    ])


def create_regression_pipeline() -> Pipeline:
    """Pipeline de regresión completo"""
    return pipe([
        node(
            func=prepare_regression_data,
            inputs=["movies", "genres"],
            outputs="regression_dataset",
            name="prepare_regression_data"
        ),
        node(
            func=train_regression_models,
            inputs="regression_dataset",
            outputs="regression_results",
            name="train_regression_models"
        ),
        node(
            func=evaluate_regression_models,
            inputs="regression_results",
            outputs="regression_metrics",
            name="evaluate_regression_models"
        )
    ])


def create_ml_modeling_pipeline() -> Pipeline:
    """Pipeline combinado que ejecuta ambos pipelines"""
    classification_pipe = create_classification_pipeline()
    regression_pipe = create_regression_pipeline()
    
    return classification_pipe + regression_pipe
