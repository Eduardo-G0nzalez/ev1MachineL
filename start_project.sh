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

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source .venv/bin/activate

# Verificar Kedro
echo "🔄 Verificando Kedro..."
kedro info
if [ $? -ne 0 ]; then
    echo "❌ Error con Kedro"
    exit 1
fi

echo ""
echo "✅ ¡Proyecto listo!"
echo ""
echo "📋 Opciones disponibles:"
echo "1. Ejecutar pipeline completo: kedro run"
echo "2. Abrir Jupyter Notebooks: kedro jupyter notebook"
echo "3. Ver información del proyecto: kedro info"
echo "4. Ejecutar tests: pytest"
echo ""

# Mantener la sesión activa
exec bash
