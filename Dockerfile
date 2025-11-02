# Dockerfile for Machine Learning Project
# Multi-stage build para optimizar tamaño

# =============================================================================
# STAGE 1: Build stage
# =============================================================================
FROM python:3.10-slim as builder

# Instalar dependencias de sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# STAGE 2: Runtime stage
# =============================================================================
FROM python:3.10-slim

# Metadatos
LABEL maintainer="Mathias Jara & Eduardo Gonzalez"
LABEL description="Machine Learning Pipeline with Kedro, DVC, Airflow"

# Crear usuario no-root
RUN useradd -m -u 1000 kedro && \
    mkdir -p /app && \
    chown kedro:kedro /app

# Establecer usuario
USER kedro

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY --chown=kedro:kedro . .

# Copiar Python packages desde builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    KEDRO_ENV=base

# Exponer puerto para Airflow (si se usa)
EXPOSE 8080

# Comando por defecto
CMD ["kedro", "run"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import kedro; print('Kedro OK')" || exit 1

