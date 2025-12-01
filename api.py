from fastapi import FastAPI
from pydantic import BaseModel
import pickle

from model import vectorizer

model = pickle.load(open(r"C:\Users\srine\Downloads\Machine learning learning\model.pkl", "rb"))
vectorizer = pickle.load(open(r"C:\Users\srine\Downloads\Machine learning learning\vectorizer.pkl" , "rb"))

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(message: Message):
    vector = vectorizer.transform([message.text])
    prediction = model.predict(vector)[0]

    return {"prediction": "spam" if prediction==1 else "ham"}