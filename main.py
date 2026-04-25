#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# -----------------------------
 
MODEL_PATH = "nlp_model_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# -----------------------------
app = FastAPI(title="NLP Prediction API")

# -----------------------------

class InputText(BaseModel):
    text: str

# -----------------------------

@app.post("/predict")
def predict(data: InputText):

    prediction = model.predict([data.text])[0]
    return {"prediction": prediction}

# -----------------------------

if __name__ == "__main__":
    test_text = "hello"
    prediction = model.predict([test_text])[0]
    print("Test Input:", test_text)
    print("Prediction:", prediction)






