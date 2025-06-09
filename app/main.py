from fastapi import FastAPI
from app.predict import IrisFeatures, predict_species

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Prediction API is running"}

@app.post("/predict")
def predict(features: IrisFeatures):
    return predict_species(features)