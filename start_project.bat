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

REM Verificar si uv está instalado
uv --version >nul 2>&1
if errorlevel 1 (
    echo ❌ uv no está instalado
    echo 🔄 Instalando uv...
    pip install uv
    if errorlevel 1 (
        echo ❌ Error instalando uv
        pause
        exit /b 1
    )
)

REM Activar entorno virtual
echo 🔄 Activando entorno virtual...
call .venv\Scripts\activate

REM Verificar Kedro
echo 🔄 Verificando Kedro...
uv run kedro info
if errorlevel 1 (
    echo ❌ Error con Kedro
    pause
    exit /b 1
)

echo.
echo ✅ ¡Proyecto listo!
echo.
echo 📋 Opciones disponibles:
echo 1. Ejecutar pipeline completo: uv run kedro run
echo 2. Abrir Jupyter Notebooks: uv run jupyter notebook --notebook-dir=notebooks --port=8888
echo 3. Ver información del proyecto: uv run kedro info
echo 4. Ejecutar tests: uv run pytest
echo.

REM Mantener la ventana abierta
cmd /k
