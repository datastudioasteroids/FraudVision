# fraud_model.py
# coding: utf-8
"""
Script para entrenamiento de modelo de detección de fraude y serialización.
Genera backend/app/model_complete.pkl con Pipeline y feature_names.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carga de datos
DATA_PATH = Path(__file__).parent / "AIML Dataset.csv"  # Ajusta nombre si difiere
if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encuentra el dataset en {DATA_PATH}")

print("Cargando dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

# 2. Ingeniería de features
# Eliminamos columnas irrelevantes
df = df.drop(columns=['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], errors='ignore')
# Creamos diferencias de balance
df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

# Definimos variables objetivo y predictoras
y = df['isFraud']
X = df.drop(columns=['isFraud'])

# 3. Columnas numéricas y categóricas
numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'balanceDiffOrig', 'balanceDiffDest']
cat_cols     = ['type']

# 4. División train/test
print("Dividiendo datos en train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Split: {X_train.shape[0]} train, {X_test.shape[0]} test")

# 5. Preprocesamiento y pipeline
# Configuramos OneHotEncoder para ignorar categorías desconocidas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='drop'
)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# 6. Entrenamiento
print("Entrenando pipeline...")
pipeline.fit(X_train, y_train)
print("Entrenamiento completado.")

# 7. Evaluación
y_pred = pipeline.predict(X_test)
print("\n===== EVALUACIÓN =====")
print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy en test: {pipeline.score(X_test, y_test):.4f}\n")

# 8. Serialización del pipeline y feature_names
print("Serializando modelo y características...")
# Extraemos nombres de features post-transformación
num_features = numeric_cols
cat_features = list(
    pipeline.named_steps['prep']
        .named_transformers_['cat']
        .get_feature_names_out(cat_cols)
)
feature_names = num_features + cat_features

# Creamos payload completo
model_payload = {
    'pipeline': pipeline,
    'feature_names': feature_names
}

# Guardamos en backend/app
OUT_PATH = Path(__file__).parent / 'backend' / 'app' / 'model_complete.pkl'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'wb') as f:
    pickle.dump(model_payload, f)
print(f"Modelo y feature_names serializados en: {OUT_PATH}")