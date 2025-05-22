#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io
import traceback
from flask import Flask, render_template, request, jsonify

# Add parent directory to path for importing from utility modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference.predict import load_model, load_label_mapping, predict

app = Flask(__name__)

# Global variables for model and label mapping
model = None
label_mapping = None
is_keras_model = False
confidence_threshold = 0.5

def load_app_model(model_path, mapping_path):
    """Load the model and label mapping."""
    global model, label_mapping, is_keras_model
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    label_mapping = load_label_mapping(mapping_path)
    is_keras_model = model_path.endswith('.h5')
    
    print(f"Model loaded with {len(label_mapping)} classes")

def preprocess_image_data(image_data):
    """Preprocess the image data for model input."""
    try:
        # Convert base64 data to image
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 64x64
        image = image.resize((64, 64), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        print(f"Preprocessed image shape: {image_array.shape}, min: {image_array.min()}, max: {image_array.max()}")
        return image_array
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sketch():
    """Predict what the sketch is."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get the image data from the request
    image_data = request.json.get('image')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        # Preprocess the image
        processed_image = preprocess_image_data(image_data)
        
        # Make prediction
        print(f"Making prediction with model: {type(model)}")
        label, confidence, probabilities = predict(model, processed_image, label_mapping, is_keras_model)
        
        print(f"Prediction result: label={label}, confidence={confidence:.4f}")
        
        # Prepare response - convert NumPy types to Python native types
        response = {
            'label': str(label),
            'confidence': float(confidence),
            'speak': bool(confidence >= confidence_threshold)  # Convert NumPy bool_ to Python bool
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error predicting sketch: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def main():
    global confidence_threshold
    
    import argparse
    parser = argparse.ArgumentParser(description="Sketch Recognition Web App")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--label_mapping", type=str, required=True, help="Path to label mapping JSON")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, 
                       help="Confidence threshold for TTS")
    
    args = parser.parse_args()
    confidence_threshold = args.confidence_threshold
    
    # Load the model
    load_app_model(args.model_path, args.label_mapping)
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 