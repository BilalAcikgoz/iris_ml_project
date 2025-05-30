from fastapi import FastAPI
from app.model import load_model, predict_species
from app.schema import IrisFeatures, PredictionResponse

app = FastAPI()

# Load model when starting
model = load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Species Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    # Predict the species of iris based on the provided features.
    prediction = predict_species(model, features)
    return prediction