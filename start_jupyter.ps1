# Script de PowerShell para iniciar Jupyter
Write-Host "Iniciando servidor Jupyter..." -ForegroundColor Green

# Cambiar al directorio del proyecto
Set-Location $PSScriptRoot

# Instalar dependencias
Write-Host "Instalando dependencias..." -ForegroundColor Yellow
uv sync

# Configurar kernel
Write-Host "Configurando kernel..." -ForegroundColor Yellow
uv run python -m ipykernel install --user --name=ev1MachineL --display-name="Python (ev1MachineL)"

# Iniciar Jupyter
Write-Host "Iniciando Jupyter Notebook..." -ForegroundColor Green
Write-Host "Accede a: http://localhost:8888" -ForegroundColor Cyan
Write-Host "Presiona Ctrl+C para detener el servidor" -ForegroundColor Red

# Usar configuración personalizada para evitar errores de permisos
uv run jupyter notebook --config=jupyter_notebook_config.py --no-browser --port=8888 --ip=127.0.0.1 --allow-root --notebook-dir=notebooks
