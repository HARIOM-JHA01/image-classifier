from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

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
efficientdet_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
model = hub.load(efficientdet_url)

# Load COCO class names
coco_class_names_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
coco_class_names = tf.keras.utils.get_file("coco_class_names.json", coco_class_names_url)

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

# Function to get COCO category names
def get_coco_category_name(class_label):
    with open(coco_class_names, "r") as f:
        class_names = f.read().splitlines()
    return class_names[class_label]

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

        # Process the detection results
        detected_objects = []
        for i in range(int(detections["num_detections"].numpy())):
            class_label = int(detections["detection_classes"][0, i].numpy())
            confidence = float(detections["detection_scores"][0, i].numpy())
            bounding_box = detections["detection_boxes"][0, i].numpy()

            coco_category_name = get_coco_category_name(class_label)

            if confidence > 0.5:
                print(f"Detected class {coco_category_name} with confidence {confidence}")
                print("Bounding box coordinates:", bounding_box)

                detected_objects.append({
                    'class': coco_category_name,
                    'confidence': confidence,
                    'bbox': bounding_box.tolist()
                })

        num_objects = len(detected_objects)

        # Remove the temporary image
        tf.io.gfile.remove(img_path)

        return {"result": f"Detected {num_objects} objects", "objects": detected_objects}
    except Exception as e:
        return HTTPException(detail=str(e), status_code=500)
