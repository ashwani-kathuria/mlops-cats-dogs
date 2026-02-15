from fastapi import FastAPI, UploadFile, File
import shutil
import torch
from src.inference.predict import predict, load_model, preprocess_image
import time
from pydantic import BaseModel
from contextlib import asynccontextmanager


model = None

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

    temp_path = "temp.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---- Start timer ----
    start_time = time.time()

    # Preprocess + Predict
    img_tensor = preprocess_image(temp_path)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    classes = ["Cat", "Dog"]
    result = classes[pred.item()]

    # ---- End timer ----
    elapsed = (time.time() - start_time) * 1000  # milliseconds

    return {
        "prediction": result,
        "time_ms": round(elapsed, 2)
    }

