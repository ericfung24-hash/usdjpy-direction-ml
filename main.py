from fastapi import FastAPI
from src.data_loader import load_and_align_usdjpy
import os

app = FastAPI(title="USDJPY Direction Prediction API")

DATA_DIR = "data"

@app.get("/")
def root():
    return {"message": "USDJPY Direction ML Service is running on Render", "status": "ok"}

@app.get("/predict")
def predict():
    try:
        h1_path = os.path.join(DATA_DIR, "USDJPY_H1.csv")
        h4_path = os.path.join(DATA_DIR, "USDJPY_H4.csv")
        d1_path = os.path.join(DATA_DIR, "USDJPY_D1.csv")

        df = load_and_align_usdjpy(h1_path, h4_path, d1_path)
        latest = df.iloc[-1]

        # 暫時用簡單規則（之後會改成真實模型）
        direction = "UP" if latest.get("returns", 0) > 0 else "DOWN"

        return {
            "symbol": "USDJPY",
            "direction": direction,
            "timestamp": str(latest.name),
            "note": "目前為測試版，之後會接真實模型預測"
        }
    except Exception as e:
        return {"error": str(e)}
