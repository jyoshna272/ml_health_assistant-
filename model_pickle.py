import tensorflow as tf
import pickle
import numpy as np

print("Loading keras model...")
model = tf.keras.models.load_model("plant_disease_model.keras")

print("Converting to pickle...")
# Save weights and config
model_config = model.get_config()
model_weights = model.get_weights()

with open("plant_model_weights.pkl", "wb") as f:
    pickle.dump({
        "config": model_config,
        "weights": model_weights
    }, f)

print("✅ Saved as plant_model_weights.pkl!")