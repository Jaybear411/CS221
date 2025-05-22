#!/usr/bin/env python3
import os
import numpy as np
import argparse
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from time import time

def load_data(data_path):
    """Load the dataset from a .npz file."""
    data = np.load(data_path)
    return data['images'], data['labels']

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a nearest-neighbor classifier")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (.npz)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data (.npz)")
    parser.add_argument("--label_mapping", type=str, required=True, help="Path to label mapping JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--n_neighbors", type=int, default=1, help="Number of neighbors for kNN")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for training (optional)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X_train, y_train = load_data(args.train_data)
    X_test, y_test = load_data(args.test_data)
    
    # Sample training data if specified (to speed up computation)
    if args.sample_size is not None and args.sample_size < len(X_train):
        indices = np.random.choice(len(X_train), args.sample_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"Sampled {args.sample_size} instances for training")
    
    # Reshape images to be flat vectors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Training set: {X_train_flat.shape}, Test set: {X_test_flat.shape}")
    
    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)
    
    # Train nearest-neighbor model
    print(f"Training {args.n_neighbors}-NN model...")
    start_time = time()
    model = KNeighborsClassifier(n_neighbors=args.n_neighbors, n_jobs=-1)
    model.fit(X_train_flat, y_train)
    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the model
    with open(os.path.join(args.output_dir, 'oracle_nn_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Evaluate the model
    print("Evaluating model...")
    start_time = time()
    y_pred = model.predict(X_test_flat)
    inference_time = time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Oracle NN accuracy: {accuracy:.4f}")
    print(f"Inference time for {len(X_test)} samples: {inference_time:.2f} seconds")
    print(f"Average inference time per sample: {1000 * inference_time / len(X_test):.2f} ms")
    
    # Get class names
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(args.output_dir, 'oracle_nn_report.txt'), 'w') as f:
        f.write(f"Oracle NN accuracy: {accuracy:.4f}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Inference time for {len(X_test)} samples: {inference_time:.2f} seconds\n")
        f.write(f"Average inference time per sample: {1000 * inference_time / len(X_test):.2f} ms\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'oracle_nn_confusion_matrix.png'))
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 