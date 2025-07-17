# Usa Airflow oficial como base
FROM apache/airflow:2.9.0-python3.9

# Cambia al usuario airflow (¡clave!)
USER airflow

# Copia tu requirements.txt dentro del contenedor
COPY airflow/requirements.txt /requirements.txt

# Instala tus requirements como el usuario airflow
RUN pip install --no-cache-dir -r /requirements.txt
