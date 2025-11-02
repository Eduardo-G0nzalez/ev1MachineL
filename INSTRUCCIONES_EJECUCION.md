# Instrucciones de Ejecución - Fase 4

## ✅ PROYECTO LISTO PARA EJECUTAR

### Notebooks Disponibles

1. **Fase4_Clasificacion_CON_PROPOSITO.ipynb**
   - Target: Éxito comercial (Alto/Medio/Bajo)
   - 5 modelos: Logistic Regression, Decision Tree, Random Forest, KNN, SVM
   - GridSearchCV + Cross-Validation (k=5)

2. **Fase4_Regresion_CON_PROPOSITO.ipynb**
   - Target: Rating de audiencia (0.88-4.69)
   - 5 modelos: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
   - GridSearchCV + Cross-Validation (k=5)

---

## 🚀 CÓMO EJECUTAR

### Paso 1: Abrir Jupyter
```bash
cd ev1MachineL/notebooks
jupyter notebook
```

### Paso 2: Ejecutar Notebooks
1. Abre **Fase4_Clasificacion_CON_PROPOSITO.ipynb**
2. Click: **Kernel > Restart & Run All**
3. Espera resultados (≈5-10 minutos)
4. Repite con **Fase4_Regresion_CON_PROPOSITO.ipynb**

---

## ✅ Checklist Pre-Ejecución

- [x] Datos disponibles en `data/01_raw/`
- [x] Notebooks completos y verificados
- [x] Librerías importadas correctamente
- [x] 5 modelos por notebook
- [x] GridSearchCV + Cross-Validation configurado
- [x] Métricas apropiadas definidas

---

## 📊 Resultados Esperados

### Clasificación
- Accuracy: ≥75%
- Mejor modelo: Random Forest o Gradient Boosting
- Confusion matrices generadas
- Gráficos comparativos

### Regresión
- R²: ≥0.65
- Mejor modelo: Random Forest o Gradient Boosting
- Métricas: R², MAE, RMSE
- Gráficos de predicciones

---

## 🎯 Propósito de las Hipótesis

### Clasificación: Éxito Comercial
**¿Qué responde?** "¿Será exitosa comercialmente esta película?"
**Para quién:** Distribuidoras, marketing, decisiones de inversión

### Regresión: Rating de Audiencia
**¿Qué responde?** "¿Qué calificación recibirá del público?"
**Para quién:** Plataformas streaming, adquisiciones de contenido

---

**AUTORES**: 
- Mathias Jara - Full Stack Developer
- Eduardo Gonzalez - Data Scientist

