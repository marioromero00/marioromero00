import pandas as pd
import os
#extraer ka data y caso que no se ecuentra
def extract_data(data_dir, filename="transaccion.parquet"):

    path = os.path.join(data_dir, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"no se encontró el archivo en la ruta: {path}")
    
    df = pd.read_parquet(path)
    return df
 