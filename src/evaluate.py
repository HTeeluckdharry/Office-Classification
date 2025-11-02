import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Snippet of train.py.
# Get result without train

# --- Configuration ---
DATA_DIR = '../dataset'  # Set to your main data directory
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Your specific classes (must be in the same order as training)
CLASSES = ['glass', 'keyboard', 'mouse', 'notebook', 'pen', 'stapler']

# Model parameters (must match training)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 1. Load Test Data ---
print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=CLASSES,
    shuffle=False  # Important: Do not shuffle test data
)

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 2. Load Trained Model ---
print("Loading pre-trained model...")
try:
    model = load_model('../office_goods_classifier.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'office_goods_classifier.keras' is in the same directory.")
    exit()

print("Model loaded successfully.")
model.summary()

# --- 3. Evaluation on Test Set (The code you wanted to run) ---
print("\n--- Test Set Evaluation ---")

# Get test loss and accuracy
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions and true labels
y_true_list = []
y_pred_list = []

print("Running predictions on test set...")
for images, labels in test_dataset:
    y_true_list.extend(labels.numpy())
    # This line is FIXED from your previous error
    y_pred_list.extend(model.predict_on_batch(images))

# Convert from one-hot/probabilities to class indices
y_true = np.argmax(y_true_list, axis=1)
y_pred = np.argmax(y_pred_list, axis=1)

# 3a. Classification Report (Accuracy, F1-Macro, F1-Weighted)
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# 3b. Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(cm)
