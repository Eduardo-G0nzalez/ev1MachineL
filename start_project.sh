#!/bin/bash

echo "🎬 Iniciando proyecto Letterboxd Machine Learning"
echo "================================================"

# Verificar si existe el entorno virtual
if [ ! -d ".venv" ]; then
    echo "❌ Entorno virtual no encontrado"
    echo "🔄 Ejecutando configuración inicial..."
    python3 setup.py
    if [ $? -ne 0 ]; then
        echo "❌ Error en la configuración"
        exit 1
    fi
fi

# Verificar si uv está instalado
if ! command -v uv &> /dev/null; then
    echo "❌ uv no está instalado"
    echo "🔄 Instalando uv..."
    pip install uv
    if [ $? -ne 0 ]; then
        echo "❌ Error instalando uv"
        exit 1
    fi
fi

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source .venv/bin/activate

# Verificar Kedro
echo "🔄 Verificando Kedro..."
uv run kedro info
if [ $? -ne 0 ]; then
    echo "❌ Error con Kedro"
    exit 1
fi

echo ""
echo "✅ ¡Proyecto listo!"
echo ""
echo "📋 Opciones disponibles:"
echo "1. Ejecutar pipeline completo: uv run kedro run"
echo "2. Abrir Jupyter Notebooks: uv run jupyter notebook --notebook-dir=notebooks --port=8888"
echo "3. Ver información del proyecto: uv run kedro info"
echo "4. Ejecutar tests: uv run pytest"
echo ""

# Mantener la sesión activa
exec bash
