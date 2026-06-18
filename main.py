from fastapi import FastAPI

app = FastAPI(
    title="USDJPY Direction Prediction Service",
    description="USDJPY 方向預測 API（Render 部署版）",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "USDJPY Direction ML Service is running on Render",
        "status": "ok"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
