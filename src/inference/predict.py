#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import argparse
import json
import pickle
from PIL import Image
import io
import base64
import sys

# Add parent directory to path for importing from utility modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tts.speak import speak_text, format_message

def load_model(model_path):
    """Load a model from file, supporting both Keras and scikit-learn formats."""
    if model_path.endswith('.h5'):
        # Load Keras model
        return tf.keras.models.load_model(model_path)
    else:
        # Load scikit-learn model
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def load_label_mapping(mapping_path):
    """Load the label mapping from a JSON file."""
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    return mapping

def preprocess_image(image_data, size=(64, 64)):
    """Preprocess an image for model input."""
    # Convert to grayscale
    img = image_data.convert('L')
    
    # Resize to the expected input size
    img = img.resize(size, Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    return img_array

def predict_keras(model, image, label_mapping):
    """Make a prediction with a Keras model."""
    try:
        print(f"Image shape before reshaping: {image.shape}, dtype: {image.dtype}")
        
        # Reshape image for CNN (add batch and channel dimensions)
        image_tensor = image.reshape(1, image.shape[0], image.shape[1], 1)
        print(f"Reshaped tensor shape: {image_tensor.shape}")
        
        # Get prediction probabilities
        print("Making prediction with model...")
        pred_probs = model.predict(image_tensor)[0]
        print(f"Prediction probabilities shape: {pred_probs.shape}")
        print(f"Probabilities: {pred_probs}")
        
        # Get top prediction
        pred_idx = np.argmax(pred_probs)
        confidence = pred_probs[pred_idx]
        
        # Get label name
        label = label_mapping.get(str(pred_idx), f"Unknown-{pred_idx}")
        print(f"Predicted class index: {pred_idx}, label: {label}, confidence: {confidence}")
        
        return label, confidence, pred_probs
    except Exception as e:
        print(f"Error in predict_keras: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def predict_sklearn(model, image, label_mapping):
    """Make a prediction with a scikit-learn model."""
    # Flatten image for sklearn model
    image_flat = image.reshape(1, -1)
    
    # Get prediction
    pred_idx = model.predict(image_flat)[0]
    
    # Get confidence if available
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(image_flat)[0][pred_idx]
    else:
        confidence = 1.0  # Default confidence
    
    # Get label name
    label = label_mapping.get(str(pred_idx), f"Unknown-{pred_idx}")
    
    return label, confidence, None

def predict(model, image, label_mapping, is_keras_model):
    """Make a prediction with the given model."""
    if is_keras_model:
        return predict_keras(model, image, label_mapping)
    else:
        return predict_sklearn(model, image, label_mapping)

def main():
    parser = argparse.ArgumentParser(description="Make predictions on a sketch image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--label_mapping", type=str, required=True, help="Path to label mapping JSON")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument("--image_base64", type=str, help="Base64 encoded image data")
    parser.add_argument("--output_json", type=str, help="Path to save output JSON")
    parser.add_argument("--speak", action="store_true", help="Speak the prediction")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for speaking")
    
    args = parser.parse_args()
    
    # Check if either image_path or image_base64 is provided
    if args.image_path is None and args.image_base64 is None:
        parser.error("Either --image_path or --image_base64 must be provided")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    is_keras_model = args.model_path.endswith('.h5')
    
    # Load label mapping
    label_mapping = load_label_mapping(args.label_mapping)
    
    # Load and preprocess image
    if args.image_path:
        print(f"Processing image: {args.image_path}")
        with Image.open(args.image_path) as img:
            processed_image = preprocess_image(img)
    else:
        print("Processing base64 encoded image")
        image_data = base64.b64decode(args.image_base64)
        img = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(img)
    
    # Make prediction
    label, confidence, probabilities = predict(model, processed_image, label_mapping, is_keras_model)
    
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
    
    # Prepare output
    result = {
        "label": label,
        "confidence": float(confidence)
    }
    
    # Include probabilities if available (for Keras models)
    if probabilities is not None:
        # Convert to a list of (label, probability) pairs and sort by probability
        probs_with_labels = [(label_mapping.get(str(i), f"Unknown-{i}"), float(p)) 
                            for i, p in enumerate(probabilities)]
        probs_with_labels.sort(key=lambda x: x[1], reverse=True)
        
        # Format as a dictionary for better JSON output
        result["all_probabilities"] = {label: prob for label, prob in probs_with_labels}
    
    # Save output if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_json}")
    
    # Print result as JSON to stdout
    print(json.dumps(result, indent=2))
    
    # Speak the result if requested
    if args.speak and confidence >= args.confidence_threshold:
        message = format_message(label, confidence)
        print(f"Speaking: {message}")
        speak_text(message)
    
    return result

if __name__ == "__main__":
    main() 