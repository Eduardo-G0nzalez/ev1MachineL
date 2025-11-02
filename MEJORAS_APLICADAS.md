# Mejoras Aplicadas al Notebook de Regresión

## 🚀 OBJETIVO: Elevar R² de 0.28 a > 0.50

### ✅ Mejoras Implementadas

#### 1. **Feature Engineering Agresivo** (15+ nuevas features)

**Features originales mejoradas:**
- `minute_log`: Logaritmo de duración (normalización)
- `minute_sqrt`: Raíz cuadrada de duración
- `minute_squared`: Cuadrado de duración (captura no-linealidad)
- `date_squared`: Cuadrado del año

**Features temporales:**
- `decade_encoded`: Década codificada (2005, 2015)
- `years_from_2000`: Años transcurridos desde 2000
- `is_recent`: Binaria (1 si >= 2010, 0 si <)
- `minute_per_year`: Ratio duración/año

**Features de diversidad:**
- `genre_diversity`: Cantidad de géneros por película
- `duration_category`: Categorías de duración (1-4)
- `is_multi_genre`: Binaria (más de 3 géneros)
- `is_long_film`: >120 min
- `is_very_long`: >150 min
- `is_short`: <90 min

**Interacciones complejas:**
- `duration_year_interaction`: Duración × Año
- `duration_genre_interaction`: Duración × Géneros
- `year_genre_interaction`: Año × Géneros
- `long_recent`: Película larga Y reciente
- `short_old`: Película corta Y antigua
- `multi_genre_recent`: Multi-género Y reciente

#### 2. **Modelos Mejorados**

**Antes:**
- Linear Regression, Random Forest básico
- Hiperparámetros limitados

**Ahora:**
- Random Forest: 300-500 árboles, max_depth 25-30
- Extra Trees: 300-500 árboles, max_depth 25-30
- Gradient Boosting: 300-500 árboles, max_depth 10-15, learning_rate fino
- XGBoost (si disponible): tuning agresivo + early stopping
- Ensemble final: combinación de mejores modelos

#### 3. **Optimizaciones de Procesamiento**

- **n_jobs=-1**: Usa todos los cores del CPU
- **tree_method='hist'** en XGBoost: Más rápido
- **Filtrado de outliers**: Solo películas 30-300 min
- **CV=5**: Validación cruzada robusta

#### 4. **GridSearchCV Mejorado**

**Rangos expandidos:**
- n_estimators: [300, 500] (antes [100, 200])
- max_depth: [25, 30, None] (antes [15, 20])
- learning_rate: [0.03, 0.05, 0.1] (fino)
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0] (solo XGBoost)

---

## ⏱️ Tiempo Estimado de Ejecución

**Configuración actual:**
- 3-5 modelos base con GridSearch
- + XGBoost si disponible
- + Ensemble final
- **Total: 6-8 horas** (puede variar según CPU)

**Por modelo individual:**
- Random Forest/Gradient Boosting: ~2-3 horas
- Extra Trees: ~2-3 horas
- XGBoost: ~1-2 horas
- Ensemble final: ~30 min

---

## 📊 Resultados Esperados

### R² Esperado: **0.50 - 0.65**
- Con las nuevas features: +0.10 a +0.15 de R²
- Con modelos más profundos: +0.05 a +0.10 de R²
- **Total esperado: ~0.48-0.65 R²**

### Mejoras en RMSE
- **Antes**: ~0.35
- **Ahora esperado**: **0.28-0.32**

---

## 🎯 Cómo Usar

### Opción 1: Jupyter (Recomendado - ver resultados en tiempo real)

```bash
cd ev1MachineL/notebooks
jupyter notebook
# Abre Fase4_Regresion.ipynb
# Ejecuta todas las celdas (Kernel → Run All)
```

### Opción 2: Docker + Airflow (Automatizado - para dejarlo toda la noche)

```bash
cd ev1MachineL

# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f kedro-pipeline

# Acceder a Airflow UI
# http://localhost:8080
```

**En Airflow:**
1. Abre `kedro_regression` DAG
2. Trigger DAG manualmente
3. Deja correr toda la noche
4. Revisa resultados al día siguiente

---

## ⚠️ Notas Importantes

1. **CPU Intensivo**: Los modelos usan todos los cores
2. **RAM**: Necesitarás al menos 8GB RAM disponible
3. **Tiempo**: Dejar toda la noche es lo ideal
4. **Resultados**: Se guardan automáticamente en `data/06_models/`

---

## 📝 Qué Revisar al Dia Siguiente

1. **Métricas finales en celda 11**
2. **Gráficos de predicciones vs reales (celda 15)**
3. **Mejor modelo identificado**
4. **R² esperado: > 0.50**

¡Listo para dejar corriendo toda la noche! 🌙



