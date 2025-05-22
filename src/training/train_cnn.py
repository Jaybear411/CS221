#!/usr/bin/env python3
import os
import numpy as np
import argparse
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data(data_path):
    """Load the dataset from a .npz file."""
    data = np.load(data_path)
    return data['images'], data['labels']

def create_cnn_model(input_shape, num_classes, dropout_rate=0.5, l2_reg=0.001):
    """Create a CNN model for sketch classification."""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape,
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate/2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate/2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_history(history, output_path):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (.npz)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data (.npz)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data (.npz)")
    parser.add_argument("--label_mapping", type=str, required=True, help="Path to label mapping JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--l2_reg", type=float, default=0.001, help="L2 regularization coefficient")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"cnn_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    print("Loading data...")
    X_train, y_train = load_data(args.train_data)
    X_val, y_val = load_data(args.val_data)
    X_test, y_test = load_data(args.test_data)
    
    # Reshape images for CNN (add channel dimension)
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_val = X_val.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)
    
    num_classes = len(label_mapping)
    print(f"Number of classes: {num_classes}")
    
    # Convert labels to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)
    
    # Create and compile model
    model = create_cnn_model((64, 64, 1), num_classes, args.dropout_rate, args.l2_reg)
    
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs')
        )
    ]
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train_onehot,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val_onehot),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the trained model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # Plot training history
    plot_history(history, os.path.join(output_dir, 'training_history.png'))
    
    # Evaluate on test set
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test, num_classes))
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get class names
    class_names = [label_mapping[str(i)] for i in range(num_classes)]
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(output_dir, 'test_report.txt'), 'w') as f:
        f.write(f"Test accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n\n")
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 