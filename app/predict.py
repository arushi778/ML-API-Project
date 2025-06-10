from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model/model.pkl")

class StudyHours(BaseModel):
   hours: float

def predict_pass_fail(features: StudyHours):
    data = np.array([[features.hours]])
    prediction = model.predict(data)
    result = int(prediction[0])
    return {"prediction": result}