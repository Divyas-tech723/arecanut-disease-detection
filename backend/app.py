from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()
# ✅ Allow frontend (HTML page) to connect to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] if using Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = r"C:\Users\divya\OneDrive\Desktop\arecanut-disease-detection\training\cnn_model.h5"
model = load_model(MODEL_PATH)

# Define your class labels
CLASS_NAMES = ["Healthy_Leaf", "Leaf Spot Disease", "yellow leaf disease"]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # must match training size
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        preds = model.predict(image)
        best_index = np.argmax(preds)
        confidence = float(np.max(preds))  # ✅ convert to normal float
        return JSONResponse(content={
            "class_name": CLASS_NAMES[best_index],
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)