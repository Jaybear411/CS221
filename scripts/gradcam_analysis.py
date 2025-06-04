#!/usr/bin/env python3
"""
Grad-CAM Analysis Script for Sketch Classification Model

This script provides comprehensive Grad-CAM analysis capabilities including:
- Single image analysis with visualization
- Batch analysis of multiple images
- Top-k prediction analysis
- Class-specific analysis
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from visualization.gradcam import GradCAM, load_model_for_gradcam


def analyze_single_image(model_path, image_path, label_mapping_path, output_dir, layer_name=None):
    """Analyze a single image with Grad-CAM."""
    print(f"Analyzing image: {image_path}")
    
    # Load model and data
    model = load_model_for_gradcam(model_path)
    
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L').resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 64, 64, 1)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, layer_name=layer_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate main prediction visualization
    result = gradcam.visualize_prediction(
        image_array,
        class_names=class_names,
        save_path=os.path.join(output_dir, "gradcam_prediction.png"),
        show_plot=False
    )
    
    # Analyze top predictions
    top_results = gradcam.analyze_top_predictions(
        image_array,
        class_names=class_names,
        top_k=min(5, len(class_names)),
        save_dir=os.path.join(output_dir, "top_predictions")
    )
    
    # Save analysis summary
    summary = {
        'image_path': image_path,
        'model_path': model_path,
        'predicted_class': result['predicted_class'],
        'predicted_label': class_names[result['predicted_class']],
        'confidence': float(result['confidence']),
        'top_predictions': [
            {
                'rank': r['rank'],
                'class_name': r['class_name'],
                'confidence': float(r['confidence'])
            }
            for r in top_results
        ]
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete! Results saved in {output_dir}")
    print(f"Predicted: {summary['predicted_label']} (confidence: {summary['confidence']:.3f})")
    
    return summary


def analyze_batch_images(model_path, images_dir, label_mapping_path, output_dir, layer_name=None):
    """Analyze multiple images with Grad-CAM."""
    print(f"Batch analyzing images from: {images_dir}")
    
    # Load model and data
    model = load_model_for_gradcam(model_path)
    
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        import glob
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} images to analyze")
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, layer_name=layer_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    batch_results = []
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('L').resize((64, 64))
            image_array = np.array(image) / 255.0
            image_array = image_array.reshape(1, 64, 64, 1)
            
            # Create individual output directory
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            img_output_dir = os.path.join(output_dir, f"image_{i:03d}_{img_name}")
            os.makedirs(img_output_dir, exist_ok=True)
            
            # Generate visualization
            result = gradcam.visualize_prediction(
                image_array,
                class_names=class_names,
                save_path=os.path.join(img_output_dir, "gradcam_prediction.png"),
                show_plot=False
            )
            
            # Store result
            batch_results.append({
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'predicted_class': result['predicted_class'],
                'predicted_label': class_names[result['predicted_class']],
                'confidence': float(result['confidence']),
                'output_dir': img_output_dir
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            batch_results.append({
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'error': str(e)
            })
    
    # Save batch summary
    batch_summary = {
        'total_images': len(image_files),
        'successful_analyses': len([r for r in batch_results if 'error' not in r]),
        'failed_analyses': len([r for r in batch_results if 'error' in r]),
        'results': batch_results
    }
    
    with open(os.path.join(output_dir, 'batch_analysis_summary.json'), 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    print(f"\nBatch analysis complete!")
    print(f"Successfully analyzed: {batch_summary['successful_analyses']}/{batch_summary['total_images']} images")
    print(f"Results saved in {output_dir}")
    
    return batch_summary


def analyze_specific_class(model_path, image_path, class_name, label_mapping_path, output_dir, layer_name=None):
    """Analyze how the model sees a specific class in an image."""
    print(f"Analyzing class '{class_name}' in image: {image_path}")
    
    # Load model and data
    model = load_model_for_gradcam(model_path)
    
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    # Find class index
    class_idx = None
    for idx, name in label_mapping.items():
        if name.lower() == class_name.lower():
            class_idx = int(idx)
            break
    
    if class_idx is None:
        print(f"Class '{class_name}' not found in label mapping!")
        print(f"Available classes: {list(label_mapping.values())}")
        return
    
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L').resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 64, 64, 1)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, layer_name=layer_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualization for specific class
    result = gradcam.visualize_prediction(
        image_array,
        class_names=class_names,
        target_class=class_idx,
        save_path=os.path.join(output_dir, f"gradcam_{class_name.replace(' ', '_')}.png"),
        show_plot=False
    )
    
    # Also generate for predicted class for comparison
    pred_result = gradcam.visualize_prediction(
        image_array,
        class_names=class_names,
        save_path=os.path.join(output_dir, "gradcam_predicted_class.png"),
        show_plot=False
    )
    
    # Save analysis
    analysis = {
        'image_path': image_path,
        'target_class': class_name,
        'target_class_index': class_idx,
        'target_class_confidence': float(result['confidence']),
        'predicted_class': class_names[pred_result['predicted_class']],
        'predicted_class_index': pred_result['predicted_class'],
        'predicted_class_confidence': float(pred_result['confidence'])
    }
    
    with open(os.path.join(output_dir, f'class_analysis_{class_name.replace(" ", "_")}.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Class analysis complete! Results saved in {output_dir}")
    print(f"Target class '{class_name}' confidence: {analysis['target_class_confidence']:.3f}")
    print(f"Predicted class '{analysis['predicted_class']}' confidence: {analysis['predicted_class_confidence']:.3f}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Analysis for Sketch Classification")
    parser.add_argument("--model", required=True, help="Path to trained model (.h5)")
    parser.add_argument("--labels", required=True, help="Path to label mapping JSON file")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--layer", help="Specific convolutional layer name to use")
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis commands')
    
    # Single image analysis
    single_parser = subparsers.add_parser('single', help='Analyze a single image')
    single_parser.add_argument("--image", required=True, help="Path to input image")
    
    # Batch analysis
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple images')
    batch_parser.add_argument("--images_dir", required=True, help="Directory containing images")
    
    # Class-specific analysis
    class_parser = subparsers.add_parser('class', help='Analyze specific class')
    class_parser.add_argument("--image", required=True, help="Path to input image")
    class_parser.add_argument("--class_name", required=True, help="Name of class to analyze")
    
    args = parser.parse_args()
    
    if args.command == 'single':
        analyze_single_image(args.model, args.image, args.labels, args.output, args.layer)
    elif args.command == 'batch':
        analyze_batch_images(args.model, args.images_dir, args.labels, args.output, args.layer)
    elif args.command == 'class':
        analyze_specific_class(args.model, args.image, args.class_name, args.labels, args.output, args.layer)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 