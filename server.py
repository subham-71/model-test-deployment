from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/opt/ml/model/birdnet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_input(extracted_features):
    # Pad or truncate to match the expected input size (144000)
    if extracted_features.shape[1] < 144000:
        extracted_features = np.pad(extracted_features, ((0, 0), (0, 144000 - extracted_features.shape[1])), mode='constant')
    elif extracted_features.shape[1] > 144000:
        extracted_features = extracted_features[:, :144000]

    # Reshape if necessary to match the model's input shape
    extracted_features = np.reshape(extracted_features, (1, 144000))
    return extracted_features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['instances'], dtype=np.float32)
    preprocessed_input = preprocess_input(input_data)
    
    interpreter.set_tensor(input_details[0]['index'], preprocessed_input)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return jsonify({"predictions": output_data.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
