from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array

import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Allow cross-origin requests from frontend

# Load the trained model
model = load_model('models/model.h5', compile=False)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        result, confidence = predict_tumor(file_path)

        return jsonify({
            'result': result,
            'confidence': f"{confidence*100:.2f}%",
            'image_url': f"http://127.0.0.1:5000/uploads/{file.filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
