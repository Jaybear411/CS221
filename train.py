#!/usr/bin/env python3
import os
import argparse
import subprocess
import json
from datetime import datetime

def run_command(command, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n{description}")
    
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True)
    
    # Stream the output
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to finish and get the exit code
    exit_code = process.wait()
    
    if exit_code != 0:
        print(f"Command failed with exit code {exit_code}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Train sketch recognition models")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                       help="Directory containing raw Quick Draw data")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                       help="Directory for processed data")
    parser.add_argument("--models_dir", type=str, default="models",
                       help="Directory for saving models")
    parser.add_argument("--skip_preprocessing", action="store_true",
                       help="Skip the preprocessing step if data is already processed")
    parser.add_argument("--skip_baseline", action="store_true",
                       help="Skip training the baseline models")
    parser.add_argument("--skip_cnn", action="store_true",
                       help="Skip training the CNN model")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for CNN training")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs for CNN training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate for CNN training")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                       help="Dropout rate for CNN training")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.models_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the training parameters
    with open(os.path.join(output_dir, "train_params.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Step 1: Data preprocessing
    if not args.skip_preprocessing:
        # First convert raw data to PNG images if needed
        png_dir = os.path.join(args.processed_dir, "png")
        os.makedirs(png_dir, exist_ok=True)
        
        if os.path.exists(args.data_dir) and any(f.endswith('.ndjson') for f in os.listdir(args.data_dir)):
            convert_cmd = (
                f"python data/utils/convert_to_png.py "
                f"--input_dir {args.data_dir} "
                f"--output_dir {png_dir} "
                f"--max_samples 5000"
            )
            if not run_command(convert_cmd, "Converting raw data to PNG images"):
                return
        else:
            print(f"No .ndjson files found in {args.data_dir}. Skipping conversion.")
        
        # Preprocess PNG images into train/val/test sets
        preprocess_cmd = (
            f"python data/utils/preprocess.py "
            f"--input_dir {png_dir} "
            f"--output_dir {args.processed_dir} "
            f"--val_size 0.15 "
            f"--test_size 0.15"
        )
        if not run_command(preprocess_cmd, "Preprocessing data for training"):
            return
    else:
        print("Skipping preprocessing step as requested.")
    
    # Check if we have the required files for training
    required_files = [
        os.path.join(args.processed_dir, "train.npz"),
        os.path.join(args.processed_dir, "val.npz"),
        os.path.join(args.processed_dir, "test.npz"),
        os.path.join(args.processed_dir, "label_mapping.json")
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Required file {file} not found. Make sure preprocessing step was completed.")
            return
    
    # Step 2: Train baseline model
    if not args.skip_baseline:
        baseline_dir = os.path.join(output_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        
        baseline_cmd = (
            f"python src/training/baseline.py "
            f"--train_data {args.processed_dir}/train.npz "
            f"--test_data {args.processed_dir}/test.npz "
            f"--label_mapping {args.processed_dir}/label_mapping.json "
            f"--output_dir {baseline_dir}"
        )
        if not run_command(baseline_cmd, "Training baseline classifier"):
            return
        
        # Train the oracle nearest-neighbor model (can be slow)
        oracle_dir = os.path.join(output_dir, "oracle_nn")
        os.makedirs(oracle_dir, exist_ok=True)
        
        oracle_cmd = (
            f"python src/training/oracle_nn.py "
            f"--train_data {args.processed_dir}/train.npz "
            f"--test_data {args.processed_dir}/test.npz "
            f"--label_mapping {args.processed_dir}/label_mapping.json "
            f"--output_dir {oracle_dir} "
            f"--n_neighbors 1 "
            f"--sample_size 5000"  # Use a sample to speed up training
        )
        if not run_command(oracle_cmd, "Training oracle nearest-neighbor classifier"):
            return
    else:
        print("Skipping baseline models as requested.")
    
    # Step 3: Train CNN model
    if not args.skip_cnn:
        cnn_dir = os.path.join(output_dir, "cnn")
        os.makedirs(cnn_dir, exist_ok=True)
        
        cnn_cmd = (
            f"python src/training/train_cnn.py "
            f"--train_data {args.processed_dir}/train.npz "
            f"--val_data {args.processed_dir}/val.npz "
            f"--test_data {args.processed_dir}/test.npz "
            f"--label_mapping {args.processed_dir}/label_mapping.json "
            f"--output_dir {cnn_dir} "
            f"--batch_size {args.batch_size} "
            f"--epochs {args.epochs} "
            f"--learning_rate {args.learning_rate} "
            f"--dropout_rate {args.dropout_rate}"
        )
        if not run_command(cnn_cmd, "Training CNN model"):
            return
    else:
        print("Skipping CNN model as requested.")
    
    print("\nTraining process completed successfully!")
    print(f"Results saved to {output_dir}")
    
    # Create a symlink to the latest training run
    latest_link = os.path.join(args.models_dir, "latest")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    
    try:
        # Use relative path for the symlink target
        os.symlink(os.path.basename(output_dir), latest_link)
        print(f"Created symlink {latest_link} -> {os.path.basename(output_dir)}")
    except Exception as e:
        print(f"Could not create symlink: {e}")

if __name__ == "__main__":
    main() 