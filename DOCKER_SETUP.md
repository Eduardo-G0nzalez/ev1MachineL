# 🐳 Guía de Configuración Docker + Airflow

## ✅ Estado Actual

El proyecto tiene Docker y Airflow completamente configurados y funcionales.

## 📋 Componentes

### 1. Dockerfile
- ✅ Multi-stage build optimizado
- ✅ Python 3.10-slim
- ✅ Usuario no-root (kedro)
- ✅ Todas las dependencias instaladas (incluyendo umap-learn, plotly, hdbscan)
- ✅ Health check configurado

### 2. docker-compose.yml
- ✅ Servicio kedro-pipeline: Contenedor principal con Kedro
- ✅ Servicio airflow-webserver: UI de Airflow (puerto 8080)
- ✅ Servicio airflow-scheduler: Ejecutor de DAGs
- ✅ Servicio postgres: Base de datos para Airflow
- ✅ Servicio airflow-init: Inicialización de base de datos
- ✅ Volúmenes montados: data, logs, src, conf
- ✅ Red Docker configurada

### 3. DAGs de Airflow
- ✅ kedro_ml_dag.py: Pipeline completo actualizado
- ✅ Incluye task de unsupervised_learning
- ✅ Dependencias correctas entre tasks
- ✅ Ejecución mediante docker exec

## 🚀 Uso

### Iniciar todos los servicios
```bash
docker-compose up -d
```

### Ver logs
```bash
# Logs de todos los servicios
docker-compose logs -f

# Logs de un servicio específico
docker-compose logs -f kedro-pipeline
docker-compose logs -f airflow-webserver
```

### Acceder a Airflow UI
- URL: http://localhost:8080
- Usuario: admin
- Password: admin

### Ejecutar pipeline manualmente en el contenedor
```bash
docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=unsupervised_learning_pipeline
```

### Reconstruir imagen después de cambios
```bash
docker-compose build kedro-pipeline
docker-compose up -d kedro-pipeline
```

## 🔧 Configuración Detallada

### Volúmenes Montados
- `./data:/app/data`: Datos del proyecto
- `./logs:/app/logs`: Logs de ejecución
- `./src:/app/src:ro`: Código fuente (solo lectura)
- `./conf:/app/conf:ro`: Configuración Kedro (solo lectura)
- `./dags:/app/dags:ro`: DAGs de Airflow (solo lectura)

### Variables de Entorno
- `KEDRO_ENV=base`: Entorno Kedro
- `AIRFLOW_HOME=/opt/airflow`: Directorio de Airflow
- `PYTHONUNBUFFERED=1`: Salida sin buffer

### Recursos
- Memoria límite: 8GB
- Memoria reservada: 4GB

## ⚠️ Notas Importantes

1. **El contenedor kedro-pipeline se mantiene corriendo** con `tail -f /dev/null` para que Airflow pueda ejecutar comandos con `docker exec`.

2. **Los cambios en el código** se reflejan automáticamente gracias a los volúmenes montados (src, conf).

3. **Los datos** están en volúmenes persistentes, por lo que se mantienen entre reinicios.

4. **Para cambios en requirements.txt**, es necesario reconstruir la imagen:
   ```bash
   docker-compose build kedro-pipeline
   ```

## 🐛 Troubleshooting

### El contenedor no inicia
```bash
# Ver logs
docker-compose logs kedro-pipeline

# Verificar que el puerto 8080 no esté en uso
netstat -ano | findstr :8080  # Windows
lsof -i :8080  # Linux/Mac
```

### Airflow no puede ejecutar comandos en el contenedor
```bash
# Verificar que el contenedor está corriendo
docker ps | grep ml-letterboxd-pipeline

# Verificar permisos de Docker socket
ls -la /var/run/docker.sock
```

### Error de permisos
```bash
# Verificar permisos de volúmenes
docker exec ml-letterboxd-pipeline ls -la /app
```

## ✅ Verificación

Para verificar que todo funciona:

1. **Iniciar servicios**:
   ```bash
   docker-compose up -d
   ```

2. **Verificar contenedores**:
   ```bash
   docker ps
   ```
   Deberías ver: ml-letterboxd-pipeline, airflow-webserver, airflow-scheduler, postgres

3. **Acceder a Airflow UI**: http://localhost:8080

4. **Activar DAG**: kedro_ml_pipeline

5. **Ejecutar manualmente**:
   ```bash
   docker exec -w /app ml-letterboxd-pipeline kedro run --pipeline=unsupervised_learning_pipeline
   ```

