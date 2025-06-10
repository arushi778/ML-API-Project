from fastapi import FastAPI
from app.predict import predict_pass_fail, StudyHours

app = FastAPI()
 
@app.get("/")
def root():
    return {"message": "Prediction API is running"}

@app.post("/predict")
def predict(features: StudyHours):
    return predict_pass_fail(features)