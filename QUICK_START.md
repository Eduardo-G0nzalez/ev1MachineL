# ⚡ Inicio Rápido

## 🚀 Ejecución Rápida (5 pasos)

### 1️⃣ Navegar al proyecto
```bash
cd "C:\Users\mathi\OneDrive\Escritorio\Proyecto kedro\ev1MachineL"
```

### 2️⃣ Construir imagen Docker (solo primera vez)
```bash
docker build -t kedro-ml .
```

### 3️⃣ Iniciar servicios
```bash
docker-compose up -d
```

**⚠️ IMPORTANTE**: Después de iniciar, verifica que el **scheduler** esté corriendo:
```bash
docker-compose ps
```
Debes ver `airflow-scheduler` con estado **Up**. Si no aparece, reinicia:
```bash
docker-compose down
docker-compose up -d
```

### 4️⃣ Abrir Airflow
- URL: http://localhost:8080
- Usuario: `admin`
- Contraseña: `admin`

### 5️⃣ Activar y ejecutar DAG
- Buscar `kedro_ml_pipeline`
- Activar el toggle (ON)
- Click en ▶️ "Trigger DAG"

---

## 📊 Ver Resultados

### Métricas de Modelos
```bash
type data\06_models\classification_metrics.json
type data\06_models\regression_metrics.json
```

### Gráficos
Abrir carpeta: `data/08_reporting/`

---

## 🛑 Detener Servicios
```bash
docker-compose down
```

---

> 📖 **Para guía completa**: Ver [`GUIA_EJECUCION_COMPLETA.md`](GUIA_EJECUCION_COMPLETA.md)

