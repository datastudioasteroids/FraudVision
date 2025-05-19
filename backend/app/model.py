import pickle
from pathlib import Path
from typing import List, Dict
from sklearn.pipeline import Pipeline

# 1) Apunta al pickle que genera train_model.py
#    Asegúrate de que 'model_complete.pkl' esté justo al lado de este archivo.
MODEL_PATH = Path(__file__).parent / "model_complete.pkl"

# 2) Carga el payload con pipeline + feature_names
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

pipeline: Pipeline           = data["pipeline"]
feature_names: List[str]     = data["feature_names"]

def predict(features: Dict[str, float]) -> Dict[str, float]:
    """
    Recibe un dict con las columnas originales (incluyendo 'type') y
    delega TODO el preprocessing + predict_proba al pipeline.
    """
    import pandas as pd

    # Construyo un DataFrame de una fila con la data original
    df_input = pd.DataFrame([features])

    # Dejo que el Pipeline haga scalers + OneHot + predict
    proba = pipeline.predict_proba(df_input)[0, 1]
    pred  = pipeline.predict(df_input)[0]

    return {
        "is_fraud": bool(pred),
        "fraud_probability": float(proba)
    }