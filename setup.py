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

def activate_and_install():
    """Activa el entorno virtual e instala dependencias"""
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
    
    # Instalar dependencias
    install_cmd = f"{pip_cmd} install -r requirements.txt"
    return run_command(install_cmd, "Instalando dependencias")

def verify_kedro():
    """Verifica que Kedro esté instalado correctamente"""
    if os.name == 'nt':
        kedro_cmd = ".venv\\Scripts\\kedro"
    else:
        kedro_cmd = ".venv/bin/kedro"
    
    return run_command(f"{kedro_cmd} info", "Verificando instalación de Kedro")

def main():
    """Función principal de configuración"""
    print("🎬 Configurando proyecto Letterboxd Machine Learning")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Crear entorno virtual
    if not create_virtual_environment():
        sys.exit(1)
    
    # Instalar dependencias
    if not activate_and_install():
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
    print("   kedro run")
    print("3. Abre Jupyter Notebooks:")
    print("   kedro jupyter notebook")
    print("\n📚 Lee el README.md para más información")

if __name__ == "__main__":
    main()
