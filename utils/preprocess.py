#!/usr/bin/env python3
import os
import numpy as np
import glob
from PIL import Image
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

def load_images(data_dir):
    """Load images and labels from directory structure."""
    images = []
    labels = []
    label_to_category = {}
    
    # Get all category directories
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    categories.sort()  # Ensure consistent label mapping
    
    print(f"Found {len(categories)} categories: {categories}")
    
    for i, category in enumerate(categories):
        label_to_category[i] = category
        category_path = os.path.join(data_dir, category)
        image_files = glob.glob(os.path.join(category_path, "*.png"))
        
        print(f"Loading {len(image_files)} images for category '{category}'")
        
        for image_file in tqdm(image_files, desc=f"Loading {category}"):
            try:
                img = Image.open(image_file).convert('L')  # Convert to grayscale
                img_array = np.array(img) / 255.0  # Normalize to [0,1]
                images.append(img_array)
                labels.append(i)
            except Exception as e:
                print(f"Error loading {image_file}: {e}")
    
    return np.array(images), np.array(labels), label_to_category

def split_data(images, labels, val_size=0.15, test_size=0.15, random_state=42):
    """Split data into train, validation, and test sets."""
    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate out validation set from the remaining data
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_dataset(images, labels, output_path):
    """Save the dataset as numpy arrays."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        images=images,
        labels=labels
    )
    print(f"Saved dataset to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Quick Draw PNG images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PNG images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation set size (proportion)")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test set size (proportion)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images and labels
    print("Loading images...")
    images, labels, label_to_category = load_images(args.input_dir)
    print(f"Loaded {len(images)} images with {len(np.unique(labels))} unique classes")
    
    # Save the label mapping
    with open(os.path.join(args.output_dir, "label_mapping.json"), 'w') as f:
        json.dump(label_to_category, f, indent=2)
    
    # Split data
    print("Splitting data...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        images, labels, args.val_size, args.test_size, args.random_state
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Save datasets
    print("Saving datasets...")
    save_dataset(X_train, y_train, os.path.join(args.output_dir, "train.npz"))
    save_dataset(X_val, y_val, os.path.join(args.output_dir, "val.npz"))
    save_dataset(X_test, y_test, os.path.join(args.output_dir, "test.npz"))
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main() 