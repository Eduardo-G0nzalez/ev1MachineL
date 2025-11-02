# 📦 Guía: Subir Proyecto con DVC a GitHub

## 🎯 ¿Qué hace DVC?

DVC (Data Version Control) versiona tus **datos grandes y modelos** que Git no puede manejar.

### Cómo funciona:
1. **Git** guarda los archivos `.dvc` (metadatos pequeños, ~1KB)
2. **DVC** guarda los datos reales (pueden ser GB)
3. Al clonar el proyecto, descargas los metadatos con Git y los datos con `dvc pull`

---

## 🚀 Opción 1: DVC Local (Recomendada para empezar)

### Paso 1: Asegurar que DVC está inicializado

```bash
cd "C:\Users\mathi\OneDrive\Escritorio\Proyecto kedro\ev1MachineL"

# Verificar que DVC está inicializado
dir .dvc
```

Si ya existe `.dvc/`, está inicializado ✅

### Paso 2: Ejecutar el pipeline para generar los datos

```bash
# Opción A: Ejecutar con Docker (recomendado)
docker-compose up -d
# Luego ejecutar el DAG en Airflow: http://localhost:8080

# Opción B: Ejecutar localmente con DVC
dvc repro
```

Esto ejecutará todos los stages definidos en `dvc.yaml` y generará los outputs.

### Paso 3: Agregar archivos a DVC (si no usaste `dvc repro`)

Si ya tienes los datos generados y quieres versionarlos manualmente:

```bash
# Versionar datasets intermedios
dvc add data/02_intermediate/releases_clean.csv
dvc add data/02_intermediate/countries_clean.csv
dvc add data/02_intermediate/genres_clean.csv
dvc add data/02_intermediate/min_release.csv

# Versionar datasets primarios
dvc add data/03_primary/final_df.csv
dvc add data/03_primary/integrated_data.csv
dvc add data/03_primary/final_df_us_2000s_2010s.parquet

# Versionar datasets de ML
dvc add data/05_model_input/classification_dataset.csv
dvc add data/05_model_input/regression_dataset.csv

# Versionar modelos entrenados
dvc add data/06_models/classification_results.pkl
dvc add data/06_models/regression_results.pkl
dvc add data/06_models/classification_metrics.pkl
dvc add data/06_models/regression_metrics.pkl
```

### Paso 4: Agregar archivos `.dvc` a Git

```bash
# Agregar todos los archivos .dvc generados
git add *.dvc .dvc/ data/**/*.dvc

# También agregar el dvc.yaml actualizado
git add dvc.yaml

# Verificar qué se va a commitear
git status
```

### Paso 5: Hacer commit y push

```bash
# Commit
git commit -m "feat: Agregar versionado DVC para datos y modelos"

# Push a GitHub
git push origin main
```

### Paso 6: Verificar en GitHub

En tu repositorio deberías ver:
- ✅ Archivos `.dvc` (pequeños, ~1KB cada uno)
- ✅ `dvc.yaml` (configuración del pipeline)
- ✅ `.dvc/` (configuración de DVC)

**Los datos grandes NO estarán en GitHub**, solo los metadatos.

---

## 🌐 Opción 2: DVC con Almacenamiento Remoto (Para producción)

Si quieres que los datos también estén disponibles en la nube:

### Paso 1: Configurar un remote

**Opción A: Google Drive** (Gratis, fácil)

```bash
# 1. Crear una carpeta en Google Drive llamada "dvc-storage"
# 2. Obtener el ID de la carpeta desde la URL:
#    https://drive.google.com/drive/folders/FOLDER_ID
# 3. Configurar DVC:

dvc remote add -d myremote gdrive://FOLDER_ID

# Guardar la configuración
git add .dvc/config
git commit -m "config: Agregar remote de Google Drive para DVC"
git push origin main
```

**Opción B: Amazon S3** (Requiere cuenta AWS)

```bash
dvc remote add -d myremote s3://nombre-bucket/dvc-storage
dvc remote modify myremote credentialpath ~/.aws/credentials

git add .dvc/config
git commit -m "config: Agregar remote S3 para DVC"
git push origin main
```

**Opción C: Azure Blob Storage**

```bash
dvc remote add -d myremote azure://nombre-contenedor/dvc-storage

git add .dvc/config
git commit -m "config: Agregar remote Azure para DVC"
git push origin main
```

### Paso 2: Subir datos al remote

```bash
# Subir todos los datos versionados
dvc push
```

### Paso 3: Push de metadatos a Git

```bash
git add .dvc/
git commit -m "chore: Actualizar metadatos DVC después de push"
git push origin main
```

---

## 📥 Para alguien que clone tu repositorio

Después de clonar tu proyecto de GitHub:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Eduardo-G0nzalez/ev1MachineL.git
cd ev1MachineL

# 2. Descargar los datos con DVC
dvc pull

# 3. (Opcional) Si usaste remote, también:
# dvc pull -r myremote
```

---

## ✅ Verificar que DVC funciona

### Ver qué está versionado:

```bash
# Ver estado de los archivos versionados
dvc status

# Ver métricas versionadas
dvc metrics show

# Ver todas las versiones de un archivo
dvc list data/03_primary/
```

### Ejecutar el pipeline completo:

```bash
# Ejecutar todos los stages en orden
dvc repro

# Ejecutar solo un stage específico
dvc repro prepare
dvc repro train_classification
```

---

## 🔍 Comandos Útiles de DVC

```bash
# Ver diferencias entre versiones
dvc diff

# Verificar integridad de archivos
dvc cache dir

# Limpiar caché (liberar espacio)
dvc gc

# Ver qué archivos están en caché
dvc cache ls
```

---

## 📝 Notas Importantes

1. **Datos sensibles**: No versiones datos que contengan información personal o sensible.

2. **Tamaño de datos**: DVC funciona mejor con archivos >1MB. Archivos pequeños pueden ir directamente a Git.

3. **`.gitignore`**: Los archivos versionados con DVC deben estar en `.gitignore` (DVC los maneja automáticamente).

4. **Colaboración**: Si trabajas en equipo, TODOS deben tener DVC instalado para hacer `dvc pull`.

---

## 🎯 Recomendación para tu proyecto

**Para evaluación académica**: Usa **Opción 1 (DVC Local)**
- ✅ Simple y suficiente
- ✅ Cumple el requisito de "versionado con DVC"
- ✅ Los datos pueden quedarse locales o compartirse por otros medios

**Para producción/equipo**: Usa **Opción 2 (Remote)**
- ✅ Datos accesibles desde cualquier lugar
- ✅ Mejor para colaboración
- ✅ Backups automáticos

---

**Autores**: Mathias Jara & Eduardo Gonzalez

