from fastapi import FastAPI, UploadFile, File
import shutil
import torch
from src.inference.predict import predict, load_model
import time
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )
logger = logging.getLogger("uvicorn.error")

model = None
REQUEST_COUNT = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    print("Model loaded via lifespan startup")

    yield  # API runs here

    print("API shutting down")


app = FastAPI(lifespan=lifespan)

# ---- Response model  ----
class PredictionResponse(BaseModel):
    prediction: str
    time_ms: float

@app.get("/")
def root():
    return {"message": "Cat vs Dog Model API running"}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):

    global REQUEST_COUNT
    REQUEST_COUNT += 1

    temp_path = "temp.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info("Received prediction request")
    logger.info(f"Total requests served: {REQUEST_COUNT}")

    # ---- Start timer ----
    start_time = time.time()

    result = predict(model, temp_path)

    # ---- End timer ----
    elapsed = (time.time() - start_time) * 1000  # milliseconds

    logger.info(f"Prediction result={result}, latency={elapsed:.2f}ms")

    return {
        "prediction": result,
        "time_ms": round(elapsed, 2)
    }

