# training/model_predict.py
"""
Load saved model and run prediction for a single image filepath.
Usage:
    from training.model_predict import load_model_and_predict
    probs, labels = load_model_and_predict("training/cnn_model.h5", "path/to/image.jpg")
"""
import numpy as np
from PIL import Image
import json
import os
import tensorflow as tf

DEFAULT_LABEL_MAP = {
    "Healthy_Leaf": 0,
    "Leaf Spot Disease": 1,
    "yellow leaf disease": 2
}

def load_label_map(path=None):
    # If you created a label_map.json after training, load it. Otherwise fallback to DEFAULT_LABEL_MAP.
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # invert mapping so index -> label
    inv = {v: k for k, v in DEFAULT_LABEL_MAP.items()}
    return inv

def preprocess_image(img_path, img_size=128):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def load_model(path):
    # Load Keras model (.h5)
    return tf.keras.models.load_model(path)

def predict_image(model, img_path, img_size=128, label_map_path=None):
    X = preprocess_image(img_path, img_size=img_size)
    probs = model.predict(X)[0]  # 1D array
    idx_to_label = load_label_map(label_map_path)
    # If label_map is index->label, use it directly; if it's label->index invert it.
    if all(isinstance(k, str) for k in idx_to_label.keys()):
        # assume it's index->label as strings; convert keys
        idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    # Build results
    results = []
    for i, p in enumerate(probs):
        label = idx_to_label.get(i, str(i))
        results.append({"label": label, "probability": float(p)})
    # best
    best_idx = int(np.argmax(probs))
    best_label = idx_to_label.get(best_idx, str(best_idx))
    return {"predictions": results, "best_label": best_label, "best_probability": float(probs[best_idx])}