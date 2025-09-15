@echo off
echo Iniciando Jupyter Notebook...
cd /d "%~dp0"

echo Configurando entorno...
uv sync

echo Iniciando servidor...
uv run jupyter notebook --notebook-dir=notebooks --port=8888

pause

