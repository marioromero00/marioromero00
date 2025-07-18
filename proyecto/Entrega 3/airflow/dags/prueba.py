import pandas as pd

# Cargar el archivo preprocesado
df = pd.read_parquet("/opt/airflow/output/preprocessed_data.parquet")

# Contar cuántas filas tienen target = 1 y target = 0
conteo = df["target"].value_counts()

print("✅ Conteo de clases en target:")
print(conteo)
