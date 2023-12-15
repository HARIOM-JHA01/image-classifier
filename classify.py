from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Function to preprocess image for the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict the class of an image
def predict_image_class(img_path):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0]

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        
        # Save the image temporarily
        temp_path = 'temp_image.jpg'
        image_file.save(temp_path)
        
        # Predict the class
        predicted_class = predict_image_class(temp_path)
        
        # Remove the temporary image
        os.remove(temp_path)
        
        response_data = {
            "class": predicted_class[1],
            "probability": float(predicted_class[2])
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
