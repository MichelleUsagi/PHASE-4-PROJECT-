from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Suicide Post Detection API")
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")


class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    label: str
    probability: float

# Load the serialized pipeline
model_path = os.path.join(os.getcwd(), "models", "logreg_pipeline.pkl")
pipeline = joblib.load(model_path)

@app.get("/")
def read_root():
    return {
        "message": (
            "Welcome to the Suicide Post Detection API. "
            "POST /predict with { text: string } to classify."
        )
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Obtain probabilities for each class
    probs = pipeline.predict_proba([req.text])[0]
    idx = probs.argmax()
    label = pipeline.classes_[idx]      # e.g., "non-suicide" or "suicide"
    return PredictResponse(
        text=req.text,
        label=label,
        probability=round(float(probs[idx]), 4)
    )
