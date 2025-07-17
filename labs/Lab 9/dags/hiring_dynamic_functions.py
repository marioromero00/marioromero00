import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path
from sklearn.metrics import accuracy_score

today = datetime.now().strftime("%Y-%m-%d") 

def create_folders(**kwargs):
    execution_date = kwargs['ds'] 
    base_path = f"./dags/{execution_date}" 
    os.makedirs(base_path, exist_ok=True)
    for folder in ["raw", "splits", "models","preprocessed"]: 
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)


#Para asegurarnos que se hayan creado las carpetas
if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    create_folders(ds=today)
    print("En ./dags encuentras:", os.listdir("./dags"))
    print(f"En ./dags/{today} encuentras:", os.listdir(f"./dags/{today}"))



def load_and_merge(**kwargs):
    # Fecha de ejecución y paths
    execution_date = kwargs['ds']
    base_path = os.path.join('.', 'dags', execution_date)
    pre_dir = os.path.join(base_path, 'preprocessed')
    os.makedirs(pre_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir = Path(__file__).parent.parent 

    files = ['data_1.csv', 'data_2.csv']
    dfs = []
    for fname in files:
        path = os.path.join(raw_dir, fname)
        if not os.path.isfile(path):
            print(f"Archivo no encontrado en raíz: {path}")
            continue
        try:
            tables = pd.read_html(path)
            if tables:
                df = tables[0]
                dfs.append(df)
                print(f"Tabla leída de: {fname}")
            else:
                print(f"No se encontró ninguna tabla en {fname}")
        except Exception as e:
            print(f"Error al leer HTML {fname}: {e}")

    if not dfs:
        print("No hay datos para concatenar.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    output_html = os.path.join(pre_dir, 'merged.html')
    merged.to_html(output_html, index=False)

    if os.path.isfile(output_html):
        size = os.path.getsize(output_html)
        print(f"{output_html} ({size} bytes)")
    else:
        print(f"No se encontró el archivo generado: {output_html}")





def split_data(**kwargs):
    execution_date = kwargs['ds']
    base_path = os.path.join('.', 'dags', execution_date)
    pre_dir = os.path.join(base_path, 'preprocessed')
    split_dir = os.path.join(base_path, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    merged_path = os.path.join(pre_dir, 'merged.html')
    if not os.path.isfile(merged_path):
        print(f"Archivo merged.html no encontrado en: {merged_path}")
        return
    try:
        df = pd.read_html(merged_path)[0]
        print(f"Data cargada para split: {len(df)} filas")
    except Exception as e:
        print(f"Error al leer merged.html: {e}")
        return

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_html = os.path.join(split_dir, 'train.html')
    test_html  = os.path.join(split_dir, 'test.html')
    train.to_html(train_html, index=False)
    test.to_html(test_html,   index=False)
    #para asegurarnos que funcione bien
    print(f"Split guardado: {train_html} ({os.path.getsize(train_html)} bytes), "
          f"{test_html} ({os.path.getsize(test_html)} bytes)")




def train_model(model, target_col, **kwargs):
    ds = kwargs["ds"]
    script_folder = Path(__file__).parent
    split_dir     = script_folder / ds / "splits"
    train_html    = split_dir / "train.html"
    if not train_html.is_file():
        raise FileNotFoundError(f"train.html no encontrado en: {train_html}")

    df = pd.read_html(train_html, flavor="html5lib")[0]
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada. "
                         f"Columnas: {df.columns.tolist()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",    model),
    ])

    pipeline.fit(X, y)
    print("Pipeline entrenado")

    models_dir = script_folder / ds / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{type(model).__name__}_{ds}.joblib"
    out   = models_dir / fname
    joblib.dump(pipeline, out)
    #para asegurarnos que funcione bien
    print(f"Modelo guardado en: {out}")
    print("Contenido de models:", [p.name for p in models_dir.iterdir() if p.is_file()])




def evaluate_models(**kwargs):
    ds = kwargs['ds']
    script_folder = Path(__file__).parent
    base         = script_folder / ds
    splits_dir   = base / 'splits'
    models_dir   = base / 'models'
    test_html = splits_dir / 'test.html'
    if not test_html.is_file():
        print(f"test.html no encontrado en: {test_html}")
        return

    df_test = pd.read_html(test_html, flavor='html5lib')[0]
    cols = df_test.columns.tolist()
    target_col = cols[-1]  # como no se especifica en el enunciado asumiremos última columna como columna objetivo
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    best_acc = -1.0
    best_name = None
    best_pipe = None

    for model_file in models_dir.glob("*.joblib"):
        pipe = joblib.load(model_file)
        # usaremos accurcay en el pipeline.score
        try:
            acc = pipe.score(X_test, y_test)
        except Exception:
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

        print(f"{model_file.name}: accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_name = model_file.name
            best_pipe = pipe

    if best_pipe is None:
        print("No se encontraron modelos para evaluar.")
        return

    print(f"Mejor modelo: {best_name} con accuracy = {best_acc:.4f}")
    best_path = models_dir / f"best_model_{ds}.joblib"
    joblib.dump(best_pipe, best_path)
    # para asegurarno que esta funcionando bien vemos donde queda guardado
    print(f"modelo guardado: {best_path}")




#Prueba de todas las funciones
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    today = datetime.now().strftime("%Y-%m-%d")
    create_folders(ds=today)
    load_and_merge(ds=today)
    split_data(ds=today)
    script_folder = Path(__file__).parent
    train_html    = script_folder / today / "splits" / "train.html"
    with open(train_html, "r", encoding="utf-8") as f:
        df = pd.read_html(f, flavor="html5lib")[0]
    print("Columnas disponibles en train.html:", df.columns.tolist())
    print(df.head())
    target_col = df.columns[-1] 
    train_model(
        RandomForestClassifier(n_estimators=10, random_state=42),
        target_col=target_col,
        ds=today
    )
    evaluate_models(ds=today)












