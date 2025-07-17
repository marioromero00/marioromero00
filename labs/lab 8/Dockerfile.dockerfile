
FROM python:3.10-slim

# crear directorio de trabajo dentro del contenedor
WORKDIR /app

# copiar archivos al contenedor
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto por donde corre la API
EXPOSE 8000

# Comando para levantar el servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
