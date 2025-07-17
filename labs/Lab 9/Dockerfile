FROM apache/airflow:2.8.1-python3.10

USER root
RUN apt-get update && apt-get install -y curl && apt-get clean

USER airflow

# Copiar y ejecutar requirements
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ✅ Agregar binario necesario para share=True en Gradio
RUN mkdir -p /home/airflow/.local/lib/python3.10/site-packages/gradio/ && \
    curl -L -o /home/airflow/.local/lib/python3.10/site-packages/gradio/frpc_linux_aarch64_v0.2 \
    https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_aarch64 && \
    chmod +x /home/airflow/.local/lib/python3.10/site-packages/gradio/frpc_linux_aarch64_v0.2


# Copiar dags
COPY dags/ /opt/airflow/dags/
