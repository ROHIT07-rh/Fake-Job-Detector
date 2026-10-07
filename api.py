
import os
import json
from datetime import datetime
from typing import Optional

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Model files
MODEL_FILE = "models/fake_job_model.pkl"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"

# History file
HISTORY_FILE = "data/prediction_history.json"

# Model evaluation metrics file
METRICS_FILE = "models/metrics.json"


# Create FastAPI application
app = FastAPI(
    title="Fake Job Detection API",
    description="API for Fake Job Posting Detection",
    version="1.0.0"
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Check model files
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

if not os.path.exists(VECTORIZER_FILE):
    raise FileNotFoundError(
        f"Vectorizer file not found: {VECTORIZER_FILE}"
    )


# Load ML model and vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)


# Request data
class JobRequest(BaseModel):
    job_text: Optional[str] = None
    text: Optional[str] = None


# Prediction function
def predict(job_text):
    job_tfidf = vectorizer.transform([job_text])

    prediction = model.predict(job_tfidf)[0]
    probabilities = model.predict_proba(job_tfidf)[0]

    if prediction == 1:
        return "FAKE", float(probabilities[1] * 100)

    return "REAL", float(probabilities[0] * 100)


# Load history
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


# Save history
def save_history(history):
    os.makedirs("data", exist_ok=True)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Fake Job Detection API is running"
    }


# Prediction endpoint
@app.post("/predict")
def detect_job(request: JobRequest):

    input_text = (request.job_text or request.text or "").strip()

    if not input_text:
        return {
            "success": False,
            "message": "Job posting cannot be empty"
        }

    # ML prediction
    label, confidence = predict(input_text)

    is_fake = bool(label == "FAKE")

    result = {
        "success": True,
        "prediction": label,
        "result": "Fake Job" if is_fake else "Real Job",
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "confidence_formatted": f"{confidence:.2f}%"
    }

    # Save prediction to history
    history = load_history()

    history_item = {
        "id": len(history) + 1,
        "job_text": input_text,
        "prediction": label,
        "result": result["result"],
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    history.append(history_item)
    save_history(history)

    return result


# History endpoint
@app.get("/history")
def get_history():
    return {
        "success": True,
        "history": load_history()
    }
# Model metrics endpoint
@app.get("/metrics")
def get_metrics():
    if not os.path.exists(METRICS_FILE):
        return {
            "success": False,
            "message": "Metrics file not found"
        }

    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        return {
            "success": True,
            "metrics": metrics
        }

    except (json.JSONDecodeError, OSError):
        return {
            "success": False,
            "message": "Unable to read metrics file"
        }

