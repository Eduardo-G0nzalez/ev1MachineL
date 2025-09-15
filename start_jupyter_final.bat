@echo off
title Jupyter Notebook - Proyecto Kedro
echo ========================================
echo    INICIANDO JUPYTER NOTEBOOK
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Instalando dependencias...
uv sync > nul 2>&1

echo [2/3] Configurando kernel...
uv run python -m ipykernel install --user --name=ev1MachineL --display-name="Python (ev1MachineL)" > nul 2>&1

echo [3/3] Iniciando Jupyter Notebook...
echo.
echo ========================================
echo   SERVIDOR INICIADO EXITOSAMENTE
echo ========================================
echo.
echo URL: http://localhost:8888
echo.
echo Presiona Ctrl+C para detener el servidor
echo ========================================
echo.

uv run python -m jupyter notebook --notebook-dir=notebooks --port=8888 --ip=127.0.0.1

