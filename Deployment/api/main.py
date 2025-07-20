from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import joblib
import os

# Import SHAP explainer function
from api.shap_utils import explain_text

app = FastAPI(title="Suicide Post Detection API")
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

# -------------------
# 📦 Pydantic models
# -------------------
class PredictRequest(BaseModel):
    text: str

class Explanation(BaseModel):
    feature: str
    shap_value: float

class PredictResponse(BaseModel):
    text: str
    label: str
    probability: float
    explanations: List[Explanation]

# -------------------------
# 🔧 Load trained pipeline
# -------------------------
model_path = os.path.join(os.getcwd(), "models", "logreg_pipeline.pkl")
pipeline = joblib.load(model_path)

# -------------------------
# 🚪 Health check endpoint
# -------------------------
@app.get("/")
def read_root():
    return {
        "message": (
            "Welcome to the Suicide Post Detection API. "
            "POST /predict with { text: string } to classify."
        )
    }

# ---------------------
# 🔍 Prediction endpoint
# ---------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text
    try:
        print("DEBUG 1: Received text →", text)

        probs = pipeline.predict_proba([text])[0]
        print("DEBUG 2: Probability scores →", probs)

        idx = probs.argmax()
        raw_label = pipeline.classes_[idx]
        label = "suicide" if raw_label == 1 else "non-suicide"
        probability = round(float(probs[idx]), 4)
        print("DEBUG 3: Predicted label →", label)
        print("DEBUG 4: Confidence score →", probability)

        explanations = explain_text(text, top_k=5)
        print("DEBUG 5: SHAP explanations →", explanations)

        return PredictResponse(
            text=text,
            label=label,
            probability=probability,
            explanations=explanations
        )

    except Exception as e:
        print(" M-% ERROR in /predict:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
