import tensorflow as tf
import numpy as np
import os

print("Loading keras model...")
model = tf.keras.models.load_model("plant_disease_model.keras")

print("Saving as SavedModel...")
tf.saved_model.save(model, "saved_model_temp")

print("Converting via command line...")
result = os.system(
    'py -3.11 -m tf2onnx.convert '
    '--saved-model saved_model_temp '
    '--output plant_disease_model.onnx '
    '--opset 13 '
    '--verbose'
)

if os.path.exists("plant_disease_model.onnx"):
    size = os.path.getsize("plant_disease_model.onnx") / 1024 / 1024
    print(f"✅ Converted! File size: {size:.1f} MB")
else:
    print("❌ Conversion failed")