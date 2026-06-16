# ============================================================================================
# FILE: server.py
# ============================================================================================
import os
import uuid
from fastapi import FastAPI, UploadFile, File
from model_helper import predict

app = FastAPI(title='Vehicle Damage Detection Engine', version="1.0.0")


# 1. HUGGING FACE HEALTH CHECK ROUTE (Required to keep the server alive)
@app.get("/")
async def health_check():
    return {"status": "Engine is awake and running!"}


# 2. A simple test endpoint
@app.get("/hello")
async def hello():
    return "Hello Bhaiya"


# 3. The main inference endpoint
@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    unique_filename = f"temp_{uuid.uuid4().hex}.jpg"

    try:
        image_bytes = await file.read()
        with open(unique_filename, 'wb') as f:
            f.write(image_bytes)

        prediction, confidence = predict(unique_filename)

        return {
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)