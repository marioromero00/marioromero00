import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import joblib
import gradio as gr

def create_folders(**kwargs):
    execution_date = kwargs['ds']
    base_path = f"./dags/{execution_date}"
    os.makedirs(base_path, exist_ok=True)
    for folder in ["raw", "splits", "models"]:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

def split_data(**kwargs):
    execution_date = kwargs['ds']
    base_path = f"./dags/{execution_date}"

    data = pd.read_csv(f"{base_path}/raw/data_1.csv")

    X = data.drop(columns=["HiringDecision"])
    y = data["HiringDecision"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train.to_csv(f"{base_path}/splits/X_train.csv", index=False)
    X_test.to_csv(f"{base_path}/splits/X_test.csv", index=False)
    y_train.to_csv(f"{base_path}/splits/y_train.csv", index=False)
    y_test.to_csv(f"{base_path}/splits/y_test.csv", index=False)

def preprocess_and_train(**kwargs):
    execution_date = kwargs['ds']
    base_path = f"./dags/{execution_date}/splits"

    X_train = pd.read_csv(f"{base_path}/X_train.csv")
    X_test = pd.read_csv(f"{base_path}/X_test.csv")
    y_train = pd.read_csv(f"{base_path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{base_path}/y_test.csv").squeeze()

    categorical_cols = ["Gender", "EducationLevel", "RecruitmentStrategy"]
    numeric_cols = ["Age", "ExperienceYears", "PreviousCompanies",
                    "DistanceFromCompany", "InterviewScore", "SkillScore", "PersonalityScore"]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f" Accuracy: {acc:.4f}")
    print(f" F1-score (contratado): {f1:.4f}")

    model_path = f"./dags/{execution_date}/models/model.joblib"
    joblib.dump(pipeline, model_path)

def gradio_interface(**kwargs):
    execution_date = kwargs['ds']
    model_path = f"./dags/{execution_date}/models/model.joblib"
    pipeline = joblib.load(model_path)

    def predict(
        Age, ExperienceYears, PreviousCompanies, DistanceFromCompany,
        InterviewScore, SkillScore, PersonalityScore,
        Gender, EducationLevel, RecruitmentStrategy
    ):
        input_df = pd.DataFrame([{
            "Age": Age,
            "ExperienceYears": ExperienceYears,
            "PreviousCompanies": PreviousCompanies,
            "DistanceFromCompany": DistanceFromCompany,
            "InterviewScore": InterviewScore,
            "SkillScore": SkillScore,
            "PersonalityScore": PersonalityScore,
            "Gender": Gender,
            "EducationLevel": EducationLevel,
            "RecruitmentStrategy": RecruitmentStrategy
        }])
        prediction = pipeline.predict(input_df)[0]
        return "Contratado" if prediction == 1 else "No Contratado"

    gr.Interface(
        fn=predict,
        inputs=[
            gr.Number(label="Edad"),
            gr.Number(label="Años de experiencia"),
            gr.Number(label="Empresas anteriores"),
            gr.Number(label="Distancia de la empresa"),
            gr.Number(label="Puntaje entrevista"),
            gr.Number(label="Puntaje habilidades"),
            gr.Number(label="Puntaje personalidad"),
            gr.Dropdown(choices=["Male", "Female"], label="Género"),
            gr.Dropdown(choices=["Bachelor", "Master", "PhD"], label="Nivel Educacional"),
            gr.Dropdown(choices=["Online", "Referral", "Walk-In"], label="Estrategia de Reclutamiento")
        ],
        outputs=gr.Text(label="Decisión"),
        title="Simulador de Contratación"
    ).launch(server_name="0.0.0.0", server_port=7860, share=False)

