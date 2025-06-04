#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
    for visualizing CNN model predictions.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM with a trained model.
        
        Args:
            model: Trained Keras/TensorFlow model
            layer_name: Name of the convolutional layer to use for visualization.
                       If None, uses the last convolutional layer.
        """
        self.model = model
        
        # Ensure model is built by making a dummy prediction if needed
        try:
            if not hasattr(model, '_built') or not model._built:
                dummy_input = tf.zeros((1, 64, 64, 1))  # Assuming 64x64 input
                _ = model(dummy_input)
        except:
            pass
        
        self.layer_name = layer_name or self._get_last_conv_layer_name()
        print(f"Using layer for Grad-CAM: {self.layer_name}")
        
        # Create a model that maps inputs to the outputs of the target layer
        # as well as the output predictions using a more robust approach
        self._create_grad_model()
    
    def _create_grad_model(self):
        """Create the gradient model for Grad-CAM computation."""
        try:
            # Method 1: Try the standard approach first
            target_layer = self.model.get_layer(self.layer_name)
            self.grad_model = tf.keras.models.Model(
                inputs=self.model.input,
                outputs=[target_layer.output, self.model.output]
            )
        except Exception as e1:
            print(f"Standard approach failed: {e1}")
            try:
                # Method 2: Try building a functional model manually
                inputs = self.model.input
                x = inputs
                target_output = None
                
                # Forward pass through layers to find our target
                for layer in self.model.layers:
                    x = layer(x)
                    if layer.name == self.layer_name:
                        target_output = x
                
                if target_output is None:
                    raise ValueError(f"Could not find layer {self.layer_name}")
                
                final_output = x
                self.grad_model = tf.keras.models.Model(
                    inputs=inputs,
                    outputs=[target_output, final_output]
                )
                
            except Exception as e2:
                print(f"Manual approach also failed: {e2}")
                # Method 3: Fallback to a simpler approach using layer indexing
                conv_layers = [i for i, layer in enumerate(self.model.layers) 
                              if (hasattr(layer, 'filters') or 'conv' in layer.__class__.__name__.lower())]
                
                if not conv_layers:
                    raise ValueError("No convolutional layers found in the model")
                
                target_layer_idx = conv_layers[-1]  # Last conv layer
                target_layer = self.model.layers[target_layer_idx]
                self.layer_name = target_layer.name
                
                print(f"Fallback: Using layer index {target_layer_idx}, name: {self.layer_name}")
                
                # Create a simple wrapper that can extract intermediate outputs
                self.grad_model = self._create_extraction_model(target_layer_idx)
    
    def _create_extraction_model(self, target_layer_idx):
        """Create a model that can extract intermediate layer outputs."""
        class IntermediateModel(tf.keras.Model):
            def __init__(self, original_model, target_idx):
                super().__init__()
                self.original_model = original_model
                self.target_idx = target_idx
                
            def call(self, inputs):
                x = inputs
                target_output = None
                
                for i, layer in enumerate(self.original_model.layers):
                    x = layer(x)
                    if i == self.target_idx:
                        target_output = x
                
                return [target_output, x]
        
        return IntermediateModel(self.model, target_layer_idx)
    
    def _get_last_conv_layer_name(self):
        """Find the name of the last convolutional layer in the model."""
        conv_layers = []
        for layer in self.model.layers:
            # Check if it's a convolutional layer by type and attributes
            if (hasattr(layer, 'filters') or 
                'conv' in layer.__class__.__name__.lower() or
                'Conv' in layer.__class__.__name__):
                conv_layers.append(layer.name)
        
        if conv_layers:
            return conv_layers[-1]  # Return the last one
        
        raise ValueError("No convolutional layer found in the model")
    
    def generate_gradcam(self, image, class_idx=None):
        """
        Generate Grad-CAM heatmap for a given image.
        
        Args:
            image: Input image array (should match model input shape)
            class_idx: Index of the class for which to generate the heatmap.
                      If None, uses the predicted class.
        
        Returns:
            heatmap: Grad-CAM heatmap as numpy array
            prediction: Model prediction probabilities
            predicted_class: Index of predicted class
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get the activations of the last conv layer and the predictions
            conv_outputs, predictions = self.grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Extract the prediction for the target class
            class_output = predictions[:, class_idx]
        
        # Compute gradients of the class output with respect to feature maps
        grads = tape.gradient(class_output, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array by the corresponding gradient
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap between 0 and 1 for visualization
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy(), predictions.numpy(), int(class_idx)
    
    def create_superimposed_visualization(self, image, heatmap, alpha=0.6):
        """
        Create a superimposed visualization of the original image and heatmap.
        
        Args:
            image: Original input image
            heatmap: Grad-CAM heatmap
            alpha: Transparency factor for the heatmap overlay
        
        Returns:
            superimposed: Combined visualization as PIL Image
        """
        # Ensure image is in the right format
        if len(image.shape) == 4:
            image = image[0]  # Remove batch dimension
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]  # Remove channel dimension for grayscale
        
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # Remove alpha channel
        
        # Convert grayscale image to RGB
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image
        
        # Ensure both images are in [0, 1] range
        image_rgb = image_rgb.astype(np.float32)
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0
        
        # Superimpose the heatmap on the image
        superimposed = (1 - alpha) * image_rgb + alpha * heatmap_colored
        superimposed = np.clip(superimposed, 0, 1)
        
        # Convert to PIL Image
        superimposed_pil = Image.fromarray((superimposed * 255).astype(np.uint8))
        
        return superimposed_pil
    
    def visualize_prediction(self, image, class_names=None, save_path=None, 
                           show_plot=True, target_class=None):
        """
        Create a comprehensive visualization of the prediction with Grad-CAM.
        
        Args:
            image: Input image
            class_names: List of class names for labeling
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
            target_class: Specific class to visualize (if None, uses predicted class)
        
        Returns:
            dict: Dictionary containing heatmap, predictions, and visualization
        """
        # Generate Grad-CAM
        heatmap, predictions, predicted_class = self.generate_gradcam(
            image, class_idx=target_class
        )
        
        # Get prediction confidence
        confidence = predictions[0][predicted_class]
        
        # Create superimposed visualization
        superimposed = self.create_superimposed_visualization(image, heatmap)
        
        # Create the plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(image.shape) == 4:
            img_display = image[0]
        else:
            img_display = image
        
        if len(img_display.shape) == 3 and img_display.shape[-1] == 1:
            img_display = img_display[:, :, 0]
        
        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Superimposed
        axes[2].imshow(superimposed)
        if class_names and predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f"Class {predicted_class}"
        
        axes[2].set_title(f'Grad-CAM Overlay\n{class_name} ({confidence:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return {
            'heatmap': heatmap,
            'predictions': predictions,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'superimposed': superimposed,
            'figure': fig if not show_plot else None
        }
    
    def analyze_top_predictions(self, image, class_names=None, top_k=3, save_dir=None):
        """
        Analyze and visualize the top-k predictions with Grad-CAM.
        
        Args:
            image: Input image
            class_names: List of class names
            top_k: Number of top predictions to analyze
            save_dir: Directory to save visualizations
        
        Returns:
            list: List of analysis results for each top prediction
        """
        # Get predictions
        _, predictions, _ = self.generate_gradcam(image)
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = []
        
        for i, class_idx in enumerate(top_indices):
            confidence = predictions[0][class_idx]
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            
            print(f"\nAnalyzing {class_name} (confidence: {confidence:.3f})")
            
            # Generate visualization for this class
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"gradcam_top{i+1}_{class_name}.png")
            
            result = self.visualize_prediction(
                image, 
                class_names=class_names,
                target_class=class_idx,
                save_path=save_path,
                show_plot=False
            )
            
            result['rank'] = i + 1
            result['class_name'] = class_name
            results.append(result)
        
        return results


def load_model_for_gradcam(model_path):
    """
    Load a saved model for Grad-CAM analysis.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)


def demo_gradcam(model_path, image_path, label_mapping_path=None, output_dir="gradcam_output"):
    """
    Demonstration function for Grad-CAM usage.
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        label_mapping_path: Path to label mapping JSON file
        output_dir: Directory to save outputs
    """
    import json
    
    # Load model
    print("Loading model...")
    model = load_model_for_gradcam(model_path)
    
    # Load image
    print("Loading image...")
    image = Image.open(image_path).convert('L').resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 64, 64, 1)
    
    # Load class names if available
    class_names = None
    if label_mapping_path and os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    
    # Initialize Grad-CAM
    print("Initializing Grad-CAM...")
    gradcam = GradCAM(model)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate single prediction visualization
    result = gradcam.visualize_prediction(
        image_array,
        class_names=class_names,
        save_path=os.path.join(output_dir, "gradcam_prediction.png")
    )
    
    # Analyze top predictions
    if class_names:
        print("\nAnalyzing top predictions...")
        top_results = gradcam.analyze_top_predictions(
            image_array,
            class_names=class_names,
            top_k=min(3, len(class_names)),
            save_dir=os.path.join(output_dir, "top_predictions")
        )
    
    print(f"\nGrad-CAM analysis complete! Results saved in {output_dir}")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Grad-CAM analysis on a model and image")
    parser.add_argument("--model", required=True, help="Path to trained model (.h5)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--labels", help="Path to label mapping JSON file")
    parser.add_argument("--output", default="gradcam_output", help="Output directory")
    
    args = parser.parse_args()
    
    demo_gradcam(args.model, args.image, args.labels, args.output) 