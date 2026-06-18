from fastapi import FastAPI
import pandas as pd
from src.data_loader import load_and_align_usdjpy
from src.feature_engineering import add_technical_features
import joblib
import os

app = FastAPI(title="USDJPY Direction Prediction API")

@app.get("/")
def root():
    return {"message": "USDJPY Direction ML Service is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# 之後我們再加上 /predict 和 /train 端點
