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

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from inference.predict import preprocess_image_data, predict_keras
    from visualization.gradcam import GradCAM
    print("Successfully imported prediction modules and Grad-CAM")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the src directory is properly set up")

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
        # Decode base64 image data
        image_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(io.BytesIO(image_data)).convert('RGBA')  # Preserve alpha

        # Create a white background image and paste the sketch onto it using alpha channel as mask
        white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask

        # Convert to grayscale
        img_gray = white_bg.convert('L')

        # Resize to 64x64 (model input size)
        img_gray = img_gray.resize((64, 64), Image.LANCZOS)

        # Convert to numpy array and normalize to [0,1]
        image_array = np.array(img_gray) / 255.0

        print(f"Preprocessed image shape: {image_array.shape}, min: {image_array.min():.3f}, max: {image_array.max():.3f}, mean: {image_array.mean():.3f}")
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

@app.route('/predict_with_gradcam', methods=['POST'])
def predict_with_gradcam():
    """Make a prediction and generate Grad-CAM visualization."""
    try:
        if 'image' not in request.json:
            return jsonify({'error': 'No image data provided'}), 400

        # Preprocess the image
        image_data = request.json['image']
        processed_image = preprocess_image_data(image_data)
        
        # Make prediction
        label, confidence, probabilities = predict_keras(model, processed_image, label_mapping)
        
        # Generate Grad-CAM visualization
        gradcam = GradCAM(model)
        
        # Prepare image for Grad-CAM (add batch and channel dimensions)
        image_tensor = processed_image.reshape(1, 64, 64, 1)
        
        # Get class names
        class_names = [label_mapping.get(str(i), f"Class {i}") for i in range(len(label_mapping))]
        
        # Generate heatmap and superimposed image
        heatmap, predictions, predicted_class = gradcam.generate_gradcam(image_tensor)
        superimposed = gradcam.create_superimposed_visualization(image_tensor, heatmap)
        
        # Convert superimposed image to base64 for web display
        buffered = io.BytesIO()
        superimposed.save(buffered, format="PNG")
        superimposed_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Convert heatmap to base64 for web display
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_buffered = io.BytesIO()
        heatmap_img.save(heatmap_buffered, format="PNG")
        heatmap_b64 = base64.b64encode(heatmap_buffered.getvalue()).decode()
        
        # Get top predictions
        top_predictions = []
        if probabilities is not None:
            top_indices = np.argsort(probabilities)[-5:][::-1]  # Top 5 predictions
            for idx in top_indices:
                class_name = label_mapping.get(str(idx), f"Unknown-{idx}")
                prob = float(probabilities[idx])
                top_predictions.append({
                    'class': class_name,
                    'probability': prob,
                    'class_index': int(idx)
                })
        
        response = {
            'prediction': label,
            'confidence': float(confidence),
            'top_predictions': top_predictions,
            'gradcam_heatmap': f"data:image/png;base64,{heatmap_b64}",
            'gradcam_overlay': f"data:image/png;base64,{superimposed_b64}",
            'success': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict_with_gradcam: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/analyze_class', methods=['POST'])
def analyze_class():
    """Analyze a specific class with Grad-CAM."""
    try:
        if 'image' not in request.json or 'class_index' not in request.json:
            return jsonify({'error': 'Image data and class_index required'}), 400

        # Preprocess the image
        image_data = request.json['image']
        processed_image = preprocess_image_data(image_data)
        
        # Get target class
        target_class = int(request.json['class_index'])
        
        # Generate Grad-CAM for specific class
        gradcam = GradCAM(model)
        image_tensor = processed_image.reshape(1, 64, 64, 1)
        
        heatmap, predictions, _ = gradcam.generate_gradcam(image_tensor, class_idx=target_class)
        superimposed = gradcam.create_superimposed_visualization(image_tensor, heatmap)
        
        # Convert to base64
        buffered = io.BytesIO()
        superimposed.save(buffered, format="PNG")
        superimposed_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_buffered = io.BytesIO()
        heatmap_img.save(heatmap_buffered, format="PNG")
        heatmap_b64 = base64.b64encode(heatmap_buffered.getvalue()).decode()
        
        # Get class information
        class_name = label_mapping.get(str(target_class), f"Unknown-{target_class}")
        class_confidence = float(predictions[0][target_class])
        
        response = {
            'class_name': class_name,
            'class_index': target_class,
            'confidence': class_confidence,
            'gradcam_heatmap': f"data:image/png;base64,{heatmap_b64}",
            'gradcam_overlay': f"data:image/png;base64,{superimposed_b64}",
            'success': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze_class: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

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