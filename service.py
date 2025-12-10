# service.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.features.feature_definitions import feature_build

app = FastAPI(title="NYC Trip Duration API")

# Load model at startup
MODEL_PATH = "model.joblib"
_model_bundle = joblib.load(MODEL_PATH)
model = _model_bundle["model"]
feature_cols = _model_bundle["feature_cols"]


class TripRequest(BaseModel):
    pickup_datetime: str
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    passenger_count: int | None = 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ready", "model_loaded": MODEL_PATH is not None}


@app.post("/predict")
def predict(req: TripRequest):
    # Convert request to DataFrame
    data = pd.DataFrame(
        [
            {
                "pickup_datetime": req.pickup_datetime,
                "pickup_latitude": req.pickup_latitude,
                "pickup_longitude": req.pickup_longitude,
                "dropoff_latitude": req.dropoff_latitude,
                "dropoff_longitude": req.dropoff_longitude,
                "passenger_count": req.passenger_count,
            }
        ]
    )

    feats = feature_build(data, tag="inference")
    # Align columns to training order
    feats = feats[feature_cols]

    pred = model.predict(feats)[0]
    return {"predicted_trip_duration": float(pred)}
