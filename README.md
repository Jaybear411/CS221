# Sketch Classification Project

A comprehensive machine learning application that recognizes hand-drawn sketches with CNN models, provides audio feedback, and includes **Grad-CAM visualization** for model interpretability and explainability.

## 🚀 Features

- **🎨 Real-time Sketch Recognition**: Draw and get instant predictions
- **🧠 CNN Model Training**: Custom CNN architecture with regularization
- **🔍 Grad-CAM Visualization**: Visual explanations of model predictions
- **🗣️ Text-to-Speech**: Audio feedback with confidence scores
- **📊 Comprehensive Evaluation**: Detailed metrics and confusion matrices
- **🌐 Web Interface**: Interactive Flask-based drawing application
- **🔄 Multiple Model Support**: Both Keras CNN and scikit-learn models
- **📱 RESTful API**: JSON-based prediction endpoints
- **📈 Batch Processing**: Efficient analysis of multiple images

## 🎯 NEW: Grad-CAM Visualization

Understand what your CNN model focuses on when making predictions:

- **Single Image Analysis**: Visualize attention patterns for individual sketches
- **Batch Analysis**: Process multiple images with visualization
- **Class-Specific Analysis**: See how the model would classify any specific class  
- **Top-K Analysis**: Visualize attention for top predictions
- **Web Integration**: Real-time Grad-CAM in the browser
- **Command Line Tools**: Flexible analysis scripts

**📖 For detailed Grad-CAM usage, see [`docs/GRADCAM_USAGE.md`](docs/GRADCAM_USAGE.md)**

## 📁 Project Structure

```
├── src/
│   ├── training/           # Model training scripts (CNN, baseline)
│   ├── inference/          # Prediction and inference utilities
│   ├── evaluation/         # Model evaluation and metrics
│   ├── visualization/      # 🆕 Grad-CAM and visualization tools
│   └── tts/               # Text-to-speech functionality
├── app/                   # Flask web application with drawing interface
├── scripts/               # 🆕 Standalone Grad-CAM analysis scripts
├── utils/                 # Data preprocessing and conversion utilities
├── tests/                 # Unit tests
├── docs/                  # 🆕 Documentation and usage guides
├── data/                  # Dataset and preprocessed data
└── models/                # Trained model files
```

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd sketch-classification
pip install -r requirements.txt
```

### Download and Process Data

1. **Download Quick Draw data**:
   ```bash
   # Create directories
   mkdir -p data/raw data/processed/png
   
   # Download from Quick Draw dataset
   cd data/raw
   gsutil cp gs://quickdraw_dataset/full/simplified/cat.ndjson .
   gsutil cp gs://quickdraw_dataset/full/simplified/tree.ndjson .
   gsutil cp gs://quickdraw_dataset/full/simplified/car.ndjson .
   cd ../..
   ```

2. **Process the data**:
   ```bash
   # Convert to PNG images
   python utils/convert_to_png.py --input_dir data/raw --output_dir data/processed/png --max_samples 5000
   
   # Preprocess images for training
   python utils/preprocess.py --input_dir data/processed/png --output_dir data/processed
   ```

### Training Models

```bash
# Train CNN model
python src/training/train_cnn.py \
    --train_data data/processed/train.npz \
    --val_data data/processed/val.npz \
    --test_data data/processed/test.npz \
    --label_mapping data/processed/label_mapping.json \
    --output_dir models/

# Train baseline model
python src/training/baseline.py \
    --train_data data/processed/train.npz \
    --test_data data/processed/test.npz \
    --label_mapping data/processed/label_mapping.json \
    --output_dir models/baseline/
```

### Making Predictions

#### Command Line Prediction
```bash
python src/inference/predict.py \
    --model_path models/best_model.h5 \
    --image_path sketch.png \
    --label_mapping data/processed/label_mapping.json
```

#### With Grad-CAM Visualization
```bash
python src/inference/predict.py \
    --model_path models/best_model.h5 \
    --image_path sketch.png \
    --label_mapping data/processed/label_mapping.json \
    --gradcam \
    --output_dir gradcam_visualizations/
```

#### Grad-CAM Analysis Scripts
```bash
# Single image analysis
python scripts/gradcam_analysis.py \
    --model models/best_model.h5 \
    --labels data/processed/label_mapping.json \
    --output results/ \
    single --image sketch.png

# Batch analysis
python scripts/gradcam_analysis.py \
    --model models/best_model.h5 \
    --labels data/processed/label_mapping.json \
    --output batch_results/ \
    batch --images_dir sketches/

# Class-specific analysis
python scripts/gradcam_analysis.py \
    --model models/best_model.h5 \
    --labels data/processed/label_mapping.json \
    --output class_analysis/ \
    class --image sketch.png --class_name "cat"
```

### Web Interface

```bash
python app/app.py \
    --model_path models/best_model.h5 \
    --label_mapping data/processed/label_mapping.json \
    --port 5000
```

Visit `http://localhost:5000` to use the interactive drawing interface with:
- Real-time sketch recognition
- Grad-CAM visualizations  
- Audio feedback
- Confidence scores

## 🔬 Model Evaluation

```bash
python src/evaluation/evaluate.py \
    --model_path models/best_model.h5 \
    --test_data data/processed/test.npz \
    --label_mapping data/processed/label_mapping.json \
    --output_dir evaluation_results/
```

## 🌐 API Endpoints

The web application provides several API endpoints:

- **POST `/predict`**: Basic prediction from sketch image
- **POST `/predict_with_gradcam`**: Prediction with Grad-CAM visualization
- **POST `/analyze_class`**: Class-specific Grad-CAM analysis

Example usage:
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
    // Display visualizations: data.gradcam_heatmap, data.gradcam_overlay
});
```

## 🏗️ Architecture

### CNN Model
- Multiple convolutional layers with max pooling
- Dropout and L2 regularization for overfitting prevention
- Dense layers for final classification
- Optimized for 64x64 grayscale sketch images

### Grad-CAM Pipeline
- Gradient computation through target convolutional layers
- Heatmap generation and overlay visualization
- Support for any convolutional layer analysis
- Batch processing capabilities

### Data Pipeline
- Quick Draw vector data conversion to normalized images
- Train/validation/test splits with proper preprocessing
- Data augmentation and normalization

## 📊 Evaluation Metrics

The project provides comprehensive evaluation including:
- Accuracy, precision, recall, F1-score
- Confusion matrices with visualization
- Confidence distribution analysis
- Per-class performance metrics
- Grad-CAM attention pattern analysis

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Structure
- **Modular design** with separate training, inference, and visualization components
- **Clean interfaces** between data processing, model training, and evaluation
- **Extensible architecture** for adding new model types and visualization techniques

## 📖 Documentation

- [`docs/GRADCAM_USAGE.md`](docs/GRADCAM_USAGE.md) - Comprehensive Grad-CAM guide
- Model training parameters and hyperparameter tuning
- API documentation and integration examples
- Troubleshooting and performance optimization tips

## 🎯 Use Cases

### Educational
- Demonstrate CNN interpretability with visual explanations
- Understand model decision-making process
- Teaching tool for machine learning concepts

### Research & Development
- Model debugging and validation
- Feature importance analysis
- Comparison of different CNN architectures

### Production Applications
- Model monitoring and quality assurance
- User trust building through explainable AI
- Error analysis and model improvement

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 🙏 Acknowledgments

- Google's Quick Draw dataset for training data
- TensorFlow/Keras for deep learning framework
- Flask for web application framework
- Grad-CAM paper authors for the visualization technique

