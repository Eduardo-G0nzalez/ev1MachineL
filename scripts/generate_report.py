"""
Script para generar reporte final de experimentos.

Genera un reporte consolidado con métricas de todos los modelos,
tabla comparativa y conclusiones.

Autores: Mathias Jara & Eduardo Gonzalez
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar src al path
project_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_path / "src"))

def generate_classification_report():
    """Generar reporte de clasificación"""
    
    # Cargar métricas
    try:
        classification_metrics = pd.read_pickle(project_path / "data/06_models/classification_metrics.pkl")
    except FileNotFoundError:
        print("⚠️ Archivo de métricas no encontrado")
        return None
    
    print("\n" + "="*80)
    print("📊 REPORTE DE MODELOS DE CLASIFICACIÓN")
    print("="*80)
    print(classification_metrics.to_string(index=False))
    
    # Mejor modelo
    best_model = classification_metrics.loc[
        classification_metrics['F1_Score'].idxmax()
    ]
    
    print(f"\n🏆 MEJOR MODELO: {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1_Score']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   CV Mean: {best_model['CV_Mean']:.4f} ± {best_model['CV_Std']:.4f}")
    
    return classification_metrics


def generate_regression_report():
    """Generar reporte de regresión"""
    
    # Cargar métricas
    try:
        regression_metrics = pd.read_pickle(project_path / "data/06_models/regression_metrics.pkl")
    except FileNotFoundError:
        print("⚠️ Archivo de métricas no encontrado")
        return None
    
    print("\n" + "="*80)
    print("📊 REPORTE DE MODELOS DE REGRESIÓN")
    print("="*80)
    print(regression_metrics.to_string(index=False))
    
    # Mejor modelo (R² más alto, RMSE más bajo)
    best_model = regression_metrics.loc[
        regression_metrics['R2_Score'].idxmax()
    ]
    
    print(f"\n🏆 MEJOR MODELO: {best_model['Model']}")
    print(f"   R² Score: {best_model['R2_Score']:.4f}")
    print(f"   RMSE: {best_model['RMSE']:.4f}")
    print(f"   CV RMSE Mean: {best_model['CV_RMSE_Mean']:.4f} ± {best_model['CV_RMSE_Std']:.4f}")
    
    return regression_metrics


def save_final_report(class_metrics, reg_metrics):
    """Guardar reporte final en markdown"""
    
    report_path = project_path / "data/07_model_output/comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte Final de Experimentos - Machine Learning\n\n")
        f.write(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Clasificación
        f.write("## 📊 Modelos de Clasificación\n\n")
        if class_metrics is not None:
            f.write(class_metrics.to_markdown(index=False))
            f.write("\n\n")
            
            best_class = class_metrics.loc[class_metrics['F1_Score'].idxmax()]
            f.write(f"**Mejor Modelo**: {best_class['Model']}\n\n")
            f.write(f"- F1-Score: {best_class['F1_Score']:.4f}\n")
            f.write(f"- CV: {best_class['CV_Mean']:.4f} ± {best_class['CV_Std']:.4f}\n\n")
        
        f.write("---\n\n")
        
        # Regresión
        f.write("## 📊 Modelos de Regresión\n\n")
        if reg_metrics is not None:
            f.write(reg_metrics.to_markdown(index=False))
            f.write("\n\n")
            
            best_reg = reg_metrics.loc[reg_metrics['R2_Score'].idxmax()]
            f.write(f"**Mejor Modelo**: {best_reg['Model']}\n\n")
            f.write(f"- R² Score: {best_reg['R2_Score']:.4f}\n")
            f.write(f"- RMSE: {best_reg['RMSE']:.4f}\n")
            f.write(f"- CV RMSE: {best_reg['CV_RMSE_Mean']:.4f} ± {best_reg['CV_RMSE_Std']:.4f}\n\n")
        
        f.write("---\n\n")
        f.write("## 📝 Conclusiones\n\n")
        f.write("### Clasificación\n\n")
        f.write("- El modelo con mejor desempeño fue identificado mediante F1-Score y validación cruzada.\n")
        f.write("- Se evaluaron 5 modelos diferentes con GridSearchCV para optimización de hiperparámetros.\n\n")
        
        f.write("### Regresión\n\n")
        f.write("- El modelo con mejor capacidad predictiva fue identificado mediante R² y RMSE.\n")
        f.write("- Se aplicó validación cruzada (k=5) para robustez de las métricas.\n\n")
        
        f.write("---\n\n")
        f.write("*Generado automáticamente por el pipeline de Kedro*\n")
        f.write(f"*Mathias Jara - Full Stack Developer*\n")
        f.write(f"*Eduardo Gonzalez - Data Scientist*\n")
    
    print(f"\n✅ Reporte guardado en: {report_path}")


if __name__ == "__main__":
    print("🚀 Generando reporte final...")
    
    class_metrics = generate_classification_report()
    reg_metrics = generate_regression_report()
    
    save_final_report(class_metrics, reg_metrics)
    
    print("\n✅ Reporte final generado exitosamente")

