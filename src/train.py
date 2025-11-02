import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# --- Configuration ---
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'office_goods_classifier.keras')
CM_SAVE_PATH = os.path.join(ROOT_DIR, 'outputs', 'confusion_matrix.png')

# Your specific classes
CLASSES = ['glass', 'keyboard', 'mouse', 'notebook', 'pen', 'stapler']
NUM_CLASSES = len(CLASSES)

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.00001

# --- 1. Load Data ---
print("Loading datasets...")

# Note: Keras sorts class names alphabetically.
# Using `class_names=CLASSES` ensures the order is fixed to yours.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=CLASSES,
    shuffle=True
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=CLASSES,
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=CLASSES,
    shuffle=False
)

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 2. Build Model (Transfer Learning) ---

# Data augmentation layers
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.2), # Shiny between glass and stapler
], name='data_augmentation')

# Input preprocessing layer
preprocess_input = applications.mobilenet_v2.preprocess_input

# Base model (MobileNetV2)
base_model = applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model initially

# Build the full model
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)  # Set training=False
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

# --- 3. Initial Training (Feature Extraction) ---
print("\n--- Starting Initial Training (Feature Extraction) ---")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_dataset,
    epochs=INITIAL_EPOCHS,
    validation_data=valid_dataset
)

# --- 4. Fine-Tuning ---
print("\n--- Starting Fine-Tuning ---")

# Unfreeze the base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100  # Unfreeze the top 54 layers

# Freeze all layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile with a very low learning rate
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Continue training
total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Start from where we left off
    validation_data=valid_dataset
)

# --- 5. Save Model ---
model.save(MODEL_SAVE_PATH)
print("\nModel saved as 'office_goods_classifier.keras'")

# --- 6. Evaluation on Test Set (as required) ---
print("\n--- Test Set Evaluation ---")

# Get test loss and accuracy
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions and true labels
y_true_list = []
y_pred_list = []

for images, labels in test_dataset:
    y_true_list.extend(labels.numpy())
    y_pred_list.extend(model.predict_on_batch(images))


# Convert from one-hot/probabilities to class indices
y_true = np.argmax(y_true_list, axis=1)
y_pred = np.argmax(y_pred_list, axis=1)

# 6a. Classification Report (Accuracy, F1-Macro, F1-Weighted)
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# 6b. Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Plot and save the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(CM_SAVE_PATH)
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")