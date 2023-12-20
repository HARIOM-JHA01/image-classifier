from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS
import requests
from io import BytesIO

app = Flask(__name__)

def get_image_metadata(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        info_dict = {
            "Filename": image.filename,
            "Image Size": image.size,
            "Image Height": image.height,
            "Image Width": image.width,
            "Image Format": image.format,
            "Image Mode": image.mode,
            "Image is Animated": getattr(image, "is_animated", False),
            "Frames in Image": getattr(image, "n_frames", 1)
        }
        datetime_original = None
        exifdata = image.getexif()
        print(exifdata[306])
        datetime_original = exifdata[306]
        return info_dict, datetime_original

    except UnidentifiedImageError as e:
        return None, str(e)

@app.route('/get_image_metadata', methods=['POST'])
def get_metadata():
    data = request.get_json()

    if 'url' not in data:
        return jsonify({"error": "URL parameter is missing"}), 400

    url = data['url']

    info_dict, datetime_original = get_image_metadata(url)

    if info_dict is None:
        return jsonify({"error": "Invalid or unsupported image format"}), 400

    result = {
        "Basic Metadata": info_dict,
        "DateTimeOriginal": datetime_original
    }
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True)

