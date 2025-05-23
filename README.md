# Sketch Interpreter

A machine learning application that recognizes hand-drawn sketches and provides audio feedback.

## Setup

1. **Create a virtual environment**:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download data**:
   
   **Option 1: Direct download from Quick Draw website**
   - Visit [Quick Draw Dataset](https://quickdraw.withgoogle.com/data)
   - Download simplified drawings (NDJSON format) for categories like:
     - cat
     - tree
     - car
     - (add more as desired)
   - Save files to `data/raw/` directory
   
   **Option 2: Command line download using gsutil**
   ```bash
   # Install Google Cloud SDK if you don't have it
   # See: https://cloud.google.com/sdk/docs/install
   
   # Create raw data directory
   mkdir -p data/raw
   cd data/raw
   
   # Download specific categories
   gsutil cp gs://quickdraw_dataset/full/simplified/cat.ndjson .
   gsutil cp gs://quickdraw_dataset/full/simplified/tree.ndjson .
   gsutil cp gs://quickdraw_dataset/full/simplified/car.ndjson .
   
   cd ../..
   ```
   
   For more information on available categories, visit:
   https://github.com/googlecreativelab/quickdraw-dataset#the-raw-moderated-dataset

## Data Processing

1. **Process the data**:
   ```bash
   # Create necessary directories
   mkdir -p data/processed/png
   
   # Convert raw data to PNG images
   python utils/convert_to_png.py --input_dir data/raw --output_dir data/processed/png --max_samples 5000
   
   # Preprocess images
   python utils/preprocess.py --input_dir data/processed/png --output_dir data/processed
   ```

2. **Train the models**:
   ```bash
   # Run full training pipeline
   python train.py
   
   # Or with custom parameters
   python train.py --batch_size 128 --epochs 30
   ```

## Usage

1. **Run the web app**:
   ```bash
   # Find your model directory (replace TIMESTAMP with actual timestamp)
   python app/app.py --model_path models/training_TIMESTAMP/cnn/best_model.h5 --label_mapping data/processed/label_mapping.json --port 5002
   ```

2. **Open in browser**: Navigate to `http://localhost:5002`

3. **Draw sketches**: Draw on the canvas and click "Predict" to get classifications

## Features

- Real-time sketch recognition
- Text-to-speech output with confidence score
- Simple, intuitive drawing interface
- Supports multiple sketch categories

## Architecture

- **Data Pipeline**: Converts Quick Draw vectors to normalized images
- **Models**: Includes baseline, nearest-neighbor, and CNN classifiers
- **Web Interface**: Flask app with HTML5 Canvas for drawing
- **TTS**: Browser-based speech synthesis for results

## Project Structure
- `data/`: Dataset storage and preprocessing utilities
- `models/`: Trained model checkpoints
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `src/`: Source code for training, inference, and evaluation
- `app/`: Web application
- `tests/`: Unit tests

## Model Architecture
The sketch recognition model uses a CNN architecture with multiple convolutional layers followed by dense layers for classification.

## Evaluation
The model is evaluated on a test set with metrics including accuracy, precision, recall, and a confusion matrix.

## License
MIT

