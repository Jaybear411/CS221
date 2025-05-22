#!/usr/bin/env python3
import os
import numpy as np
import argparse
import json
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_data(data_path):
    """Load the dataset from a .npz file."""
    data = np.load(data_path)
    return data['images'], data['labels']

def load_model(model_path):
    """Load a model from file, supporting both Keras and scikit-learn formats."""
    if model_path.endswith('.h5'):
        # Load Keras model
        return tf.keras.models.load_model(model_path)
    else:
        # Load scikit-learn model
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def predict_keras(model, images):
    """Make predictions using a Keras model."""
    # Ensure images have the right shape (add channel dimension for CNNs)
    if len(images.shape) == 3:  # (samples, height, width)
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    
    # Get prediction probabilities
    y_probs = model.predict(images)
    # Get predicted classes
    y_pred = np.argmax(y_probs, axis=1)
    # Get confidence scores
    confidence = np.max(y_probs, axis=1)
    
    return y_pred, confidence

def predict_sklearn(model, images):
    """Make predictions using a scikit-learn model."""
    # Flatten images for sklearn models
    images_flat = images.reshape(images.shape[0], -1)
    
    # Get predicted classes
    y_pred = model.predict(images_flat)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_probs = model.predict_proba(images_flat)
        confidence = np.max(y_probs, axis=1)
    else:
        confidence = np.ones_like(y_pred)  # Default confidence of 1.0
    
    return y_pred, confidence

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data (.npz)")
    parser.add_argument("--label_mapping", type=str, required=True, help="Path to label mapping JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, 
                       help="Confidence threshold for predictions")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    X_test, y_test = load_data(args.test_data)
    
    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    # Make predictions
    print("Making predictions...")
    if args.model_path.endswith('.h5'):
        y_pred, confidence = predict_keras(model, X_test)
    else:
        y_pred, confidence = predict_sklearn(model, X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Get class names
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Analyze confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidence, bins=20, alpha=0.7)
    plt.axvline(x=args.confidence_threshold, color='r', linestyle='--', 
                label=f'Threshold ({args.confidence_threshold})')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'confidence_distribution.png'))
    
    # Evaluate performance with confidence threshold
    high_conf_idx = confidence >= args.confidence_threshold
    if np.sum(high_conf_idx) > 0:
        high_conf_accuracy = accuracy_score(y_test[high_conf_idx], y_pred[high_conf_idx])
        coverage = np.mean(high_conf_idx)
        print(f"\nWith confidence threshold {args.confidence_threshold}:")
        print(f"  Accuracy: {high_conf_accuracy:.4f}")
        print(f"  Coverage: {coverage:.4f} ({np.sum(high_conf_idx)} / {len(y_test)} samples)")
    else:
        print(f"\nNo predictions above confidence threshold {args.confidence_threshold}")
    
    # Save results to file
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
        
        if np.sum(high_conf_idx) > 0:
            f.write(f"With confidence threshold {args.confidence_threshold}:\n")
            f.write(f"  Accuracy: {high_conf_accuracy:.4f}\n")
            f.write(f"  Coverage: {coverage:.4f} ({np.sum(high_conf_idx)} / {len(y_test)} samples)\n\n")
        
        f.write("Classification Report:\n")
        f.write(report)
    
    # Save prediction details for further analysis
    prediction_data = {
        'true_labels': y_test.tolist(),
        'predicted_labels': y_pred.tolist(),
        'confidence': confidence.tolist(),
        'class_names': class_names
    }
    
    with open(os.path.join(args.output_dir, 'prediction_details.json'), 'w') as f:
        json.dump(prediction_data, f)
    
    print(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 