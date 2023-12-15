# Image Classification Web Service

This is a simple Flask web service for image classification using a pre-trained ResNet50 model. The service accepts images via a POST request and returns predictions about the contents of the image.

## Getting Started

These instructions will help you set up and run the web service locally for development and testing purposes.

### Prerequisites

- Python (version 3.x)
- Install dependencies:

  ```bash
  pip install Flask flask-cors tensorflow numpy gunicorn Pillow
  ```

### Running the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/HARIOM-JHA01/image-classifier.git
   cd image-classifier
   ```

2. Run the Flask application:

   ```bash
   python classify.py
   ```

   The application will be accessible at [http://localhost:3000](http://localhost:3000).

## Usage

### API Endpoint

- **POST** `/classify`

  This endpoint accepts an image file as input and returns predictions about the image.

  Example using cURL:

  ```bash
  curl -X POST -H "Content-Type: multipart/form-data" -F "image=@path/to/your/image.jpg" http://localhost:3000/classify
  ```

  Example Response:

  ```json
  {
      "class": "dog",
      "probability": 0.975
  }
  ```

### Response Format

- **Success Response:**

  ```json
  {
      "class": "predicted_class",
      "probability": 0.75
  }
  ```

- **Error Response:**

  ```json
  {
      "error": "Error message details"
  }
  ```

## Deployment

Under Progressss!

## Contributing

Feel free to contribute to the development of this project. Create a fork, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
