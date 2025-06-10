import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/student_data.csv")

X = df[['hours']]
y = df['Result']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'model/model.pkl')