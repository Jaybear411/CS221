#!/usr/bin/env python3
import os
import numpy as np
import argparse
import json
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_data(data_path):
    """Load the dataset from a .npz file."""
    data = np.load(data_path)
    return data['images'], data['labels']

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a baseline classifier")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (.npz)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data (.npz)")
    parser.add_argument("--label_mapping", type=str, required=True, help="Path to label mapping JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X_train, y_train = load_data(args.train_data)
    X_test, y_test = load_data(args.test_data)
    
    # Reshape images to be flat
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Training set: {X_train_flat.shape}, Test set: {X_test_flat.shape}")
    
    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)
    
    # Train baseline model (most frequent class)
    print("Training baseline model...")
    model = DummyClassifier(strategy='most_frequent', random_state=42)
    model.fit(X_train_flat, y_train)
    
    # Save the model
    with open(os.path.join(args.output_dir, 'baseline_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Baseline accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(args.output_dir, 'baseline_report.txt'), 'w') as f:
        f.write(f"Baseline accuracy: {accuracy:.4f}\n\n")
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
    plt.savefig(os.path.join(args.output_dir, 'baseline_confusion_matrix.png'))
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 