import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import regularizers

# Initialize the Flask application
app = Flask(__name__)

# Build the model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
predictions = Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)  # assuming 3 classes
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model (use .h5 format instead of pickle)
model.save('model.h5')

# Load the model
model = load_model('model.h5')

# Define image size and class labels (these should match what you used for training)
img_size = (256, 256)  # Input image size for the model
class_labels = {0: 'Class1', 1: 'Class2', 2: 'Class3'}  # Modify with actual class labels

# Home route to render HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define the predict function to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Open and preprocess the image
            img = Image.open(file.stream)
            img = img.resize(img_size)  # Resize the image
            img = np.array(img)  # Convert the image to a numpy array
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = img / 255.0  # Normalize the image

            # Get model prediction
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability

            # Map the predicted class to a label
            predicted_label = class_labels.get(predicted_class[0], 'Unknown')

            # Return the prediction result
            return jsonify({'prediction': predicted_label}), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
