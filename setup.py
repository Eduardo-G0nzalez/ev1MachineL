#!/usr/bin/env python3
"""
Script de configuración para el proyecto Letterboxd Machine Learning
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e.stderr}")
        return False

def check_python_version():
    """Verifica que la versión de Python sea compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True

def create_virtual_environment():
    """Crea un entorno virtual"""
    if os.path.exists(".venv"):
        print("✅ Entorno virtual ya existe")
        return True
    
    return run_command("python -m venv .venv", "Creando entorno virtual")

def install_with_uv():
    """Instala dependencias usando uv (Astra)"""
    # Verificar si uv está instalado
    uv_check = subprocess.run("uv --version", shell=True, capture_output=True, text=True)
    if uv_check.returncode != 0:
        print("❌ uv no está instalado. Instalando uv...")
        if not run_command("pip install uv", "Instalando uv"):
            return False
    
    # Usar uv sync para crear entorno virtual e instalar dependencias
    return run_command("uv sync", "Creando entorno virtual e instalando dependencias con uv")

def verify_kedro():
    """Verifica que Kedro esté instalado correctamente"""
    return run_command("uv run kedro info", "Verificando instalación de Kedro")

def main():
    """Función principal de configuración"""
    print("🎬 Configurando proyecto Letterboxd Machine Learning")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Instalar dependencias con uv
    if not install_with_uv():
        sys.exit(1)
    
    # Verificar Kedro
    if not verify_kedro():
        print("⚠️  Kedro no se pudo verificar, pero la instalación puede haber sido exitosa")
    
    print("\n" + "=" * 50)
    print("🎉 ¡Configuración completada!")
    print("\n📋 Próximos pasos:")
    print("1. Activa el entorno virtual:")
    if os.name == 'nt':
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("2. Ejecuta el proyecto:")
    print("   uv run kedro run")
    print("3. Abre Jupyter Notebooks:")
    print("   uv run jupyter notebook --notebook-dir=notebooks --port=8888")
    print("\n📚 Lee el README.md para más información")

if __name__ == "__main__":
    main()
