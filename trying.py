from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained object detection model
model = tf.saved_model.load("path/to/efficientdet_model")  # Replace with the actual path

# Function to perform image detection
def detect_objects(image_array):
    detections = model(image_array)
    return detections

# Function to preprocess image for the model
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((512, 512))  # Adjust size based on your model requirements
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/detect_objects")
async def detect_objects_route(file: UploadFile = File(...)):
    try:
        # Save the uploaded image temporarily
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as img_file:
            img_file.write(file.file.read())

        # Preprocess the image
        image_array = preprocess_image(img_path)

        # Perform object detection
        detections = detect_objects(image_array)

        # Process the detection results as needed
        # For simplicity, returning the number of detected objects
        num_objects = int(detections["num_detections"].numpy())
        
        # Remove the temporary image
        tf.io.gfile.remove(img_path)

        return {"result": f"Detected {num_objects} objects"}
    except Exception as e:
        return HTTPException(detail=str(e), status_code=500)
