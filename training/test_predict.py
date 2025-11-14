# training/test_predict.py
import os
from model_predict import load_model, predict_image

# Resolve paths relative to this script's parent folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

model_path = os.path.join(project_root, "training", "cnn_model.h5")
test_root = os.path.join(project_root, "dataset", "test")

print("Model path:", model_path)
print("Test root:", test_root)

# find first image file under dataset/test
img_path = None
for root, _, files in os.walk(test_root):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, f)
            break
    if img_path:
        break

if img_path is None:
    raise SystemExit("No test images found under dataset/test. Put one image there and retry.")

print("Using test image:", img_path)

# load model and predict
model = load_model(model_path)
res = predict_image(model, img_path, img_size=224)  # if you trained with different size, change here
print("Prediction result:")
print(res)