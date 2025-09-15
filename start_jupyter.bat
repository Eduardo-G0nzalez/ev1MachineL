@echo off
echo Iniciando servidor Jupyter...
cd /d "%~dp0"

echo Instalando dependencias...
uv sync

echo Configurando kernel...
uv run python -m ipykernel install --user --name=ev1MachineL --display-name="Python (ev1MachineL)"

echo Iniciando Jupyter Notebook...
uv run jupyter notebook --no-browser --port=8888 --ip=127.0.0.1 --allow-root --notebook-dir=notebooks

pause

