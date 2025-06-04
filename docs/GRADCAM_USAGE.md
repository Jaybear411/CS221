# Grad-CAM Usage Guide

This guide explains how to use Gradient-weighted Class Activation Mapping (Grad-CAM) with your sketch classification model to understand what the CNN focuses on when making predictions.

## What is Grad-CAM?

Grad-CAM is a technique for producing visual explanations for decisions from CNN-based models. It shows which parts of an input image are important for a particular prediction by highlighting regions that strongly influence the model's decision.

## Features

- **Single Image Analysis**: Analyze individual sketches with comprehensive visualizations
- **Batch Processing**: Process multiple images at once
- **Class-Specific Analysis**: See how the model would classify any specific class
- **Top-K Analysis**: Visualize the model's top predictions
- **Web Interface Integration**: Real-time Grad-CAM in the web app
- **Command Line Tools**: Flexible command-line interface

## Quick Start

### 1. Command Line Usage

#### Single Image Analysis
```bash
python scripts/gradcam_analysis.py \
    --model models/best_model.h5 \
    --labels data/label_mapping.json \
    --output gradcam_results \
    single --image path/to/your/sketch.png
```

#### Batch Analysis
```bash
python scripts/gradcam_analysis.py \
    --model models/best_model.h5 \
    --labels data/label_mapping.json \
    --output batch_results \
    batch --images_dir path/to/sketch/directory
```

#### Class-Specific Analysis
```bash
python scripts/gradcam_analysis.py \
    --model models/best_model.h5 \
    --labels data/label_mapping.json \
    --output class_analysis \
    class --image path/to/sketch.png --class_name "cat"
```

### 2. Python API Usage

```python
from src.visualization.gradcam import GradCAM, load_model_for_gradcam
from PIL import Image
import numpy as np

# Load model
model = load_model_for_gradcam('models/best_model.h5')

# Load and preprocess image
image = Image.open('sketch.png').convert('L').resize((64, 64))
image_array = np.array(image) / 255.0
image_tensor = image_array.reshape(1, 64, 64, 1)

# Create Grad-CAM instance
gradcam = GradCAM(model)

# Generate visualization
result = gradcam.visualize_prediction(
    image_tensor,
    class_names=['cat', 'dog', 'bird'],  # Your class names
    save_path='gradcam_output.png'
)

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### 3. Integration with Existing Prediction Script

```bash
# Make prediction with Grad-CAM visualization
python src/inference/predict.py \
    --model_path models/best_model.h5 \
    --image_path sketch.png \
    --label_mapping data/label_mapping.json \
    --gradcam \
    --output_dir gradcam_visualizations
```

### 4. Web Interface

The web app now includes Grad-CAM functionality:

1. **Standard Prediction with Grad-CAM**: 
   - Send POST request to `/predict_with_gradcam`
   - Returns prediction + heatmap + overlay images

2. **Class-Specific Analysis**:
   - Send POST request to `/analyze_class` 
   - Specify which class to analyze

Example JavaScript usage:
```javascript
// Predict with Grad-CAM
fetch('/predict_with_gradcam', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({image: base64ImageData})
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction);
    console.log('Confidence:', data.confidence);
    // Display heatmap: data.gradcam_heatmap
    // Display overlay: data.gradcam_overlay
});
```

## Understanding the Output

### Visualization Components

1. **Original Image**: Your input sketch
2. **Grad-CAM Heatmap**: Red areas show high importance, blue areas show low importance
3. **Overlay**: Superimposed heatmap on the original image

### Interpreting Results

- **Hot Spots (Red/Yellow)**: Regions the model focuses on for this prediction
- **Cold Spots (Blue/Purple)**: Regions the model ignores
- **Confidence Score**: How certain the model is about its prediction

### Common Patterns

- **Good Model**: Focuses on relevant sketch features (e.g., cat ears for cat classification)
- **Potential Issues**: Focuses on background artifacts or irrelevant features
- **Class Confusion**: Different classes show similar attention patterns

## Advanced Usage

### Custom Layer Analysis

Specify which convolutional layer to analyze:
```bash
python scripts/gradcam_analysis.py \
    --model models/best_model.h5 \
    --labels data/label_mapping.json \
    --layer conv2d_2 \
    --output layer_analysis \
    single --image sketch.png
```

### Programmatic Batch Processing

```python
from src.visualization.gradcam import GradCAM
import os
import glob

# Load model
model = load_model_for_gradcam('models/best_model.h5')
gradcam = GradCAM(model)

# Process all images in a directory
image_files = glob.glob('sketches/*.png')
results = []

for img_path in image_files:
    # Load and preprocess
    image = Image.open(img_path).convert('L').resize((64, 64))
    image_array = np.array(image) / 255.0
    image_tensor = image_array.reshape(1, 64, 64, 1)
    
    # Analyze
    result = gradcam.visualize_prediction(
        image_tensor,
        save_path=f'output/{os.path.basename(img_path)}_gradcam.png',
        show_plot=False
    )
    
    results.append({
        'file': img_path,
        'prediction': result['predicted_class'],
        'confidence': result['confidence']
    })
```

### Top-K Analysis

```python
# Analyze top 3 predictions for an image
top_results = gradcam.analyze_top_predictions(
    image_tensor,
    class_names=class_names,
    top_k=3,
    save_dir='top_predictions_output'
)

for result in top_results:
    print(f"Rank {result['rank']}: {result['class_name']} ({result['confidence']:.3f})")
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Loading Error**: Ensure your model file is a Keras .h5 file
   ```python
   # Check if model is compatible
   import tensorflow as tf
   model = tf.keras.models.load_model('your_model.h5')
   print(model.summary())
   ```

3. **Layer Not Found**: List available layers
   ```python
   for i, layer in enumerate(model.layers):
       print(f"{i}: {layer.name} - {type(layer).__name__}")
   ```

4. **Image Preprocessing Issues**: Ensure images are preprocessed correctly
   ```python
   # Correct preprocessing for 64x64 grayscale
   image = Image.open('sketch.png').convert('L').resize((64, 64))
   image_array = np.array(image) / 255.0
   image_tensor = image_array.reshape(1, 64, 64, 1)  # (batch, height, width, channels)
   ```

### Performance Tips

1. **Batch Processing**: Process multiple images together when possible
2. **Layer Selection**: Earlier layers show low-level features, later layers show high-level concepts
3. **Memory Management**: Close matplotlib figures when processing many images

## Use Cases

### Model Debugging
- Identify if model focuses on relevant features
- Detect overfitting to background artifacts
- Compare attention patterns across different classes

### Model Validation  
- Verify model reasoning aligns with human intuition
- Identify potential biases in training data
- Validate model performance on edge cases

### Educational/Explanatory
- Show users why the model made a particular prediction
- Build trust in automated sketch recognition systems
- Demonstrate CNN interpretability techniques

### Research and Development
- Compare attention patterns between different model architectures
- Analyze the effect of data augmentation on model attention
- Study how different training techniques affect model focus

## Example Output Files

After running Grad-CAM analysis, you'll get:

```
output_directory/
├── gradcam_prediction.png          # Main visualization
├── analysis_summary.json           # Prediction details
├── top_predictions/                # Top-K analysis
│   ├── gradcam_top1_cat.png
│   ├── gradcam_top2_dog.png
│   └── gradcam_top3_bird.png
└── class_analysis_cat.json        # Class-specific analysis
```

## Integration with Your Workflow

### During Model Development
```bash
# Quick check of model behavior
python scripts/gradcam_analysis.py --model models/latest_model.h5 --labels data/labels.json --output debug single --image test_sketch.png
```

### For Model Evaluation
```bash
# Analyze model on validation set
python scripts/gradcam_analysis.py --model models/final_model.h5 --labels data/labels.json --output evaluation batch --images_dir validation_sketches/
```

### In Production Monitoring
```python
# Log attention patterns for monitoring
if confidence < threshold:
    gradcam_result = generate_gradcam_analysis(image, model)
    log_attention_pattern(gradcam_result, image_id)
```

This Grad-CAM implementation provides powerful insights into your sketch classification model's decision-making process, helping you build more reliable and interpretable AI systems. 