from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array

import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the model (no compile needed for prediction)
model = load_model('models/model.h5', compile=False)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Set upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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


@app.route('/')
def index():
    return jsonify({"message": "Tumor Detection API is running."})


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

        image_url = request.host_url + 'uploads/' + file.filename

        return jsonify({
            'result': result,
            'confidence': f"{confidence * 100:.2f}%",
            'image_url': image_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
