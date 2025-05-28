from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model_1 = joblib.load("model_1.pkl")
model_2 = joblib.load("model_2.pkl")
model_3 = joblib.load("model_3.pkl")

class RetailItem(BaseModel):
    quantity: float
    price: float

@app.post("/predict/model1")
def predict_cluster_m1(item: RetailItem):
    data = np.array([[item.quantity, item.price]])
    cluster = model_1.predict(data)[0]
    return {"model": "model_1", "cluster": int(cluster)}

@app.post("/predict/model2")
def predict_cluster_m2(item: RetailItem):
    data = np.array([[item.quantity, item.price]])
    cluster = model_2.predict(data)[0]
    return {"model": "model_2", "cluster": int(cluster)}

@app.post("/predict/model3")
def predict_cluster_m3(item: RetailItem):
    data = np.array([[item.quantity, item.price]])
    cluster = model_3.predict(data)[0]
    return {"model": "model_3", "cluster": int(cluster)}
