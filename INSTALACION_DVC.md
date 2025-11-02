# 📦 Guía de Instalación de DVC

## ¿Dónde instalar DVC?

### Opción 1: Instalación Global (Recomendada para empezar) ⭐

**Instalar en tu computadora** (disponible para todos los proyectos):

```bash
pip install dvc
```

✅ **Ventajas**:
- Simple y rápido
- Disponible en cualquier proyecto
- No necesitas activar entornos virtuales

⚠️ **Desventajas**:
- Puede causar conflictos si diferentes proyectos requieren diferentes versiones de DVC

**Cuándo usar**: Si trabajas principalmente con este proyecto y no tienes problemas de versiones.

---

### Opción 2: Instalación en el Proyecto (Mejor práctica) ⭐⭐

**Instalar solo en este proyecto** (en un entorno virtual):

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 3. Instalar DVC
pip install dvc

# 4. Verificar
dvc --version
```

✅ **Ventajas**:
- Aísla dependencias del proyecto
- No afecta otros proyectos
- Evita conflictos de versiones
- Mejor para producción y colaboración

**Cuándo usar**: Si trabajas en múltiples proyectos o vas a compartir el código.

---

### Opción 3: Solo en Docker (No instalar localmente)

Si **SOLO ejecutas el proyecto con Docker**, NO necesitas instalar DVC en tu computadora.

DVC ya está incluido en `requirements.txt`, así que se instalará automáticamente en el contenedor Docker cuando ejecutes:

```bash
docker build -t kedro-ml .
```

**Cuándo usar**: Si solo vas a ejecutar el pipeline con Docker y no necesitas comandos DVC locales.

---

## 🎯 Recomendación para tu caso

### Si ejecutas el proyecto principalmente con Docker:
→ **NO instales DVC localmente** (Opción 3)

DVC funcionará dentro del contenedor Docker cuando ejecutes:
```bash
docker-compose up -d
```

### Si quieres usar comandos DVC en tu computadora (dvc repro, dvc metrics, etc.):
→ **Instala globalmente** (Opción 1) para empezar rápido

```bash
pip install dvc
```

### Si trabajas profesionalmente o compartes el proyecto:
→ **Instala en entorno virtual** (Opción 2) - mejor práctica

---

## ✅ Verificar instalación

Después de instalar:

```bash
# Verificar versión
dvc --version

# Deberías ver algo como:
# DVC version: 3.x.x
```

---

## 🔄 Desinstalar DVC (si es necesario)

```bash
# Si instalaste globalmente
pip uninstall dvc

# Si instalaste en entorno virtual
# Solo desactiva el entorno virtual o elimínalo
```

---

## 📝 Nota importante

**DVC no es estrictamente necesario** para ejecutar el proyecto con Docker.

- ✅ Puedes ejecutar todo con `docker-compose up -d` sin tener DVC instalado
- ✅ Los datos se guardan localmente en `data/`
- ⚠️ DVC solo es útil si quieres:
  - Versionar datos y modelos con Git
  - Comparar métricas entre experimentos
  - Compartir datasets grandes con el equipo

**Para la evaluación del proyecto**: Si el requisito dice "DVC versiona datos y modelos", tener el `dvc.yaml` configurado es suficiente. La instalación local es opcional.

---

**Autores**: Mathias Jara & Eduardo Gonzalez

