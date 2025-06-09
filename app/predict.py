from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model/model.pkl")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

def predict_species(features: IrisFeatures):
    data = np.array([[features.sepal_length, features.sepal_width,
                      features.petal_length, features.petal_width]])
    prediction = model.predict(data)
    species_num = int(prediction[0])
    species_name = species_map[species_num]
    return {"prediction": species_num,
            "species": species_name}