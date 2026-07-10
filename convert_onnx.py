import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

print("Loading keras model...")
model = tf.keras.models.load_model("plant_disease_model.keras")

print("Converting to ONNX...")
# Save as SavedModel first then convert
model.export("saved_model_temp")

onnx_model, _ = tf2onnx.convert.from_saved_model(
    "saved_model_temp",
    opset=13,
    output_path="plant_disease_model.onnx"
)

print("✅ Saved as plant_disease_model.onnx!")
print(f"File size: {os.path.getsize('plant_disease_model.onnx') / 1024 / 1024:.1f} MB")