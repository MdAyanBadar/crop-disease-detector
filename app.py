from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model("./model/crop_disease_model.h5")

# Load class indices from JSON file and convert keys/values to proper types
with open('./model/class_indices.json', 'r') as f:
    raw_indices = json.load(f)
    # Ensure keys are strings and values are integers
    class_indices = {k: int(v) for k, v in raw_indices.items()}

# Reverse mapping from index to class name
index_to_class = {v: k for k, v in class_indices.items()}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((128, 128))

        # Normalize image
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        
        # Debug logs
        print("Prediction array:", prediction)
        print("Sum of prediction:", np.sum(prediction))
        print("Contains NaN:", np.isnan(prediction).any())
        print("Prediction shape:", prediction.shape)

        # Handle invalid predictions
        if np.isnan(prediction).any():
            return jsonify({'error': 'Prediction result contains NaN values'}), 500

        predicted_index = int(np.argmax(prediction))
        print("Predicted index:", predicted_index)
        print("Index-to-class map:", index_to_class)

        predicted_class = index_to_class.get(predicted_index, "Unknown class")

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
