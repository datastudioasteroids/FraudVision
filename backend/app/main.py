# backend/app/main.py

import time
import json
import io
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import pytesseract
import shap
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from datetime import datetime
from pdf2image import convert_from_bytes
from PIL import UnidentifiedImageError



# Importamos predict y pipeline desde model.py
from .model import predict, pipeline, feature_names

# -------------------------
# Inicialización de SHAP
# -------------------------
try:
    explainer = shap.TreeExplainer(pipeline.named_steps['clf'])
except Exception:
    explainer = None

# -------------------------
# Base de datos SQLite en memoria
# -------------------------
conn = sqlite3.connect(':memory:', check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    '''CREATE TABLE IF NOT EXISTS predictions (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           timestamp TEXT,
           is_fraud INTEGER,
           fraud_prob REAL
       )'''
)
conn.commit()

# -------------------------
# FastAPI + CORS
# -------------------------
app = FastAPI(title="Fraud Detection & OCR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# RUTAS DE API
# -------------------------


@app.post("/predict")
async def predict_fraud(data: dict):
    # 1) Define aquí las columnas que tu modelo espera:
    numeric_cols = [
      'amount', 'oldbalanceOrg', 'newbalanceOrig',
      'oldbalanceDest', 'newbalanceDest',
      'balanceDiffOrig', 'balanceDiffDest'
    ]
    cat_cols = ['type']

    # 2) Arma el dict de features, poniendo 0.0 para numéricas que no envíes, y "" para type
    features = {}
    for col in numeric_cols:
        features[col] = float(data.get(col, 0.0))
    # Si el frontend no envía 'type', será cadena vacía
    features['type'] = data.get('type', "")

    # 3) Llama a tu función predict con el dict completo
    result = predict(features)

    # 4) Guarda en la base y compón la respuesta
    ts = datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT INTO predictions (timestamp, is_fraud, fraud_prob) VALUES (?, ?, ?)",
        (ts, int(result['is_fraud']), result['fraud_probability'])
    )
    conn.commit()
    tx_id = cursor.lastrowid
    new_point = {"timestamp": ts, "fraud_probability": result['fraud_probability']}

    return {**result, "transaction_id": tx_id, "new_point": new_point}

@app.get("/metrics")
async def metrics():
    """
    Devuelve métricas y serie histórica de avgFraudProbability por hora.
    """
    since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE timestamp >= ?", conn, params=(since,)
    )
    total = len(df)
    frauds = df['is_fraud'].sum()
    fraud_rate = frauds / total if total else 0

    df['hr'] = pd.to_datetime(df['timestamp']).dt.floor('h')
    txn_per_hour = df.groupby('hr').size().mean() if total else 0

    hist = (
        df.groupby('hr')['fraud_prob']
          .mean()
          .reset_index()
          .rename(columns={'hr': 'timestamp', 'fraud_prob': 'avgFraudProbability'})
    )

    return {
        "current": {"fraudRate": fraud_rate, "txnPerHour": txn_per_hour},
        "history": hist.to_dict(orient='records')
    }


@app.get("/stream")
async def stream():
    """
    SSE: envía nuevos puntos en tiempo real.
    """
    def event_generator():
        last_id = 0
        while True:
            rows = cursor.execute(
                "SELECT timestamp, fraud_prob FROM predictions WHERE id > ?",
                (last_id,)
            ).fetchall()
            for ts, prob in rows:
                last_id += 1
                yield f"data: {json.dumps({'timestamp': ts, 'fraud_probability': prob})}\n\n"
            time.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/batch")
async def batch(file: UploadFile = File(...)):
    """
    Predicción en lote desde CSV o Parquet.
    """
    ext = file.filename.split('.')[-1].lower()
    df = pd.read_csv(file.file) if ext == 'csv' else pd.read_parquet(file.file)
    probs = pipeline.predict_proba(df)[:, 1]
    return {"frauds_detected": int((probs > 0.5).sum())}


@app.post("/upload_ticket")
async def upload_ticket(file: UploadFile = File(...)):
    content = await file.read()
    images = []

    # 1) Si es PDF, convierto cada página a imagen
    if file.content_type == "application/pdf":
        try:
            pages = convert_from_bytes(content)
            images.extend(pages)
        except Exception as e:
            raise HTTPException(400, f"No se pudo convertir PDF a imagen: {e}")
    # 2) Si es imagen, la cargo directamente
    elif file.content_type.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(content))
            images.append(img)
        except UnidentifiedImageError:
            raise HTTPException(400, "Formato de imagen no soportado o corrupto")
    else:
        raise HTTPException(400, "Solo se aceptan archivos PDF o imágenes")

    # 3) Hago OCR sobre todas las páginas
    full_text = ""
    for img in images:
        full_text += pytesseract.image_to_string(img, lang="spa") + "\n"

    # 4) Extraigo campos y predigo
    def extract_val(text, field):
        m = re.search(rf"{field}\s*[:]?\s*(\d+[\.,]?\d*)", text)
        return float(m.group(1).replace(",", ".")) if m else 0.0

    features = {
        'amount':         extract_val(full_text, 'amount'),
        'oldbalanceOrg':  extract_val(full_text, 'oldbalanceOrg'),
        'newbalanceOrig': extract_val(full_text, 'newbalanceOrig'),
        'oldbalanceDest': extract_val(full_text, 'oldbalanceDest'),
        'newbalanceDest': extract_val(full_text, 'newbalanceDest'),
        'balanceDiffOrig':extract_val(full_text, 'balanceDiffOrig'),
        'balanceDiffDest':extract_val(full_text, 'balanceDiffDest'),
        'type':           re.search(r"type\s*[:]?\s*(\w+)", full_text).group(1) if re.search(r"type\s*[:]?\s*(\w+)", full_text) else ""
    }

    result = predict(features)
    return {
        "filename": file.filename,
        "is_fraud": result["is_fraud"],
        "fraud_probability": result["fraud_probability"]
    }

@app.post("/upload_documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    responses = []
    for file in files:
        content = await file.read()
        images = []
        if file.content_type == 'application/pdf':
            # Convertir cada página de PDF a imagen PIL
            pages = convert_from_bytes(content)
            images.extend(pages)
        else:
            images.append(Image.open(io.BytesIO(content)))

        # Concatenar OCR de todas las páginas
        raw_text = "\n".join(pytesseract.image_to_string(img, lang='spa') for img in images)

        # Extraer campos clave (uso tus funciones de regex)
        def extract_val(text, field):
            m = re.search(rf"{field}\\s*[:]?\s*(\\d+[\\.,]?\\d*)", text)
            return float(m.group(1).replace(',', '.')) if m else 0.0

        # Construir dict para predict()
        features = {
            'amount': extract_val(raw_text, 'amount'),
            'oldbalanceOrg': extract_val(raw_text, 'oldbalanceOrg'),
            'newbalanceOrig': extract_val(raw_text, 'newbalanceOrig'),
            'oldbalanceDest': extract_val(raw_text, 'oldbalanceDest'),
            'newbalanceDest': extract_val(raw_text, 'newbalanceDest'),
            'balanceDiffOrig': extract_val(raw_text, 'balanceDiffOrig'),
            'balanceDiffDest': extract_val(raw_text, 'balanceDiffDest'),
            'type': re.search(r"type\\s*[:]?\s*(\\w+)", raw_text).group(1) if re.search(...) else ''
        }

        # Llamada a tu predict()
        result = predict(features)
        responses.append({
            'filename': file.filename,
            **result
        })

    return {'results': responses}

@app.get("/features")
async def features():
    """
    Importancias de características.
    """
    clf = pipeline.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = clf.coef_[0]
    else:
        raise HTTPException(503, "No se pueden extraer importancias")

    feats = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
    return [{"name": n, "importance": float(i)} for n, i in feats]


@app.get("/shap_values")
async def shap_values(id_transaccion: int):
    """
    Valores SHAP (demo vacío si no hay datos originales).
    """
    if explainer is None:
        raise HTTPException(503, "SHAP no disponible")
    dfp = pd.read_sql_query(
        "SELECT * FROM predictions WHERE id = ?", conn, params=(id_transaccion,)
    )
    if dfp.empty:
        raise HTTPException(404, "No encontrada")
    return {"top_features": []}


# -------------------------
# Montaje del frontend (STATIC) — al final
# -------------------------
frontend = Path(__file__).parent.parent / "frontend"
app.mount(
    "/",
    StaticFiles(directory=frontend, html=True),
    name="frontend"
)


# -------------------------
# Ejecutar servidor
# -------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
