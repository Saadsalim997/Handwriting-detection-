from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, template_folder=r'C:\Users\Kratos\Handwriting Detection\templates',
            static_folder=r'C:\Users\Kratos\Handwriting Detection\static')

# Load the trained model
model_path = r'C:\Users\Kratos\Handwriting Detection\handwritten_digit_model.h5'
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = image.load_img(file, target_size=(28, 28), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])

    return jsonify({'prediction': str(predicted_class)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000), debug=True)
