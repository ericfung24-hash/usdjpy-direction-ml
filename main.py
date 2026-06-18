from fastapi import FastAPI
from src.data_loader import load_and_align_usdjpy
import pandas as pd
import os

app = FastAPI(title="USDJPY Direction Prediction API")

DATA_DIR = "data"

@app.get("/")
def root():
    return {"message": "USDJPY Direction ML Service is running", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/predict")
def predict_direction():
    """
    目前為測試版：載入最新資料並回傳方向預測
    之後會改成載入真實訓練好的模型
    """
    try:
        h1_path = os.path.join(DATA_DIR, "USDJPY_H1.csv")
        h4_path = os.path.join(DATA_DIR, "USDJPY_H4.csv")
        d1_path = os.path.join(DATA_DIR, "USDJPY_D1.csv")

        df = load_and_align_usdjpy(h1_path, h4_path, d1_path)
        latest = df.iloc[-1]

        # 暫時用簡單規則（之後改成模型預測）
        direction = "UP" if latest["returns"] > 0 else "DOWN"
        confidence = round(abs(latest["returns"]) * 100, 2)

        return {
            "symbol": "USDJPY",
            "direction": direction,
            "confidence": confidence,
            "timestamp": str(latest.name),
            "note": "目前為測試版，之後會接真實模型"
        }
    except Exception as e:
        return {"error": str(e)}
