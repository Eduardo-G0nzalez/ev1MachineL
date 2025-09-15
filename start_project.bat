@echo off
echo 🎬 Iniciando proyecto Letterboxd Machine Learning
echo ================================================

REM Verificar si existe el entorno virtual
if not exist ".venv" (
    echo ❌ Entorno virtual no encontrado
    echo 🔄 Ejecutando configuración inicial...
    python setup.py
    if errorlevel 1 (
        echo ❌ Error en la configuración
        pause
        exit /b 1
    )
)

REM Activar entorno virtual
echo 🔄 Activando entorno virtual...
call .venv\Scripts\activate

REM Verificar Kedro
echo 🔄 Verificando Kedro...
kedro info
if errorlevel 1 (
    echo ❌ Error con Kedro
    pause
    exit /b 1
)

echo.
echo ✅ ¡Proyecto listo!
echo.
echo 📋 Opciones disponibles:
echo 1. Ejecutar pipeline completo: kedro run
echo 2. Abrir Jupyter Notebooks: kedro jupyter notebook
echo 3. Ver información del proyecto: kedro info
echo 4. Ejecutar tests: pytest
echo.

REM Mantener la ventana abierta
cmd /k
