FROM python:3.8-slim

# Install TensorFlow Lite
RUN pip install tensorflow==2.8.0 flask gunicorn

# Copy the model and the script to the container
COPY ./models/birdnet.tflite /opt/ml/model/birdnet.tflite
COPY ./serve.py /opt/ml/model/serve.py

# Set the working directory
WORKDIR /opt/ml/model

# Start the Flask application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "serve:app"]
