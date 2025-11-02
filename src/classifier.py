import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os  # We need this to check if the file exists

# --- Configuration ---
MODEL_PATH = '.models/office_goods_classifier.keras'
IMG_SIZE = (224, 224)
CLASSES = ['glass', 'keyboard', 'mouse', 'notebook', 'pen', 'stapler']

# --- 1. Load Model ---
try:
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'office_goods_classifier.keras' is in the same directory.")
    exit()


def preprocess_image(frame, resize_shape):
    """
    Preprocesses a single frame for model prediction.
    """
    img = cv2.resize(frame, resize_shape)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    return img_batch


def post_process_prediction(prediction):
    """
    Takes the model's raw output and returns the class name and confidence.
    """
    pred_index = np.argmax(prediction[0])
    pred_class = CLASSES[pred_index]
    confidence = np.max(prediction[0]) * 100
    return pred_class, confidence


def classify_image(image_path):
    """
    Classifies a single image file.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return

    processed_img = preprocess_image(frame, IMG_SIZE)
    prediction = model.predict(processed_img)
    pred_class, confidence = post_process_prediction(prediction)

    label_text = f"{pred_class} ({confidence:.1f}%)"
    print(f"Prediction: {label_text}")

    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image Classification Result", frame)
    print("Press any key to close the image.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def classify_camera():
    """
    Classifies video from the default camera in real-time.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera feed opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        processed_frame = preprocess_image(frame.copy(), IMG_SIZE)
        prediction = model.predict(processed_frame)
        pred_class, confidence = post_process_prediction(prediction)

        label_text = f"{pred_class} ({confidence:.1f}%)"
        color = (0, 255, 0) if confidence > 60 else (0, 0, 255)

        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Classification - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- 2. Main execution logic (CHANGED) ---
if __name__ == "__main__":
    print("Office Goods Classifier")
    print("-----------------------")
    print("Select mode:")
    print("  [1] Live Camera")
    print("  [2] Single Image File")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        print("\nStarting live camera feed...")
        classify_camera()

    elif choice == '2':
        print("\nPlease provide the path to your image.")
        print("💡 TIP: Drag and drop your image file onto this window, then press Enter.")
        image_path = input("Image Path: ")

        # Clean up the path (drag-and-drop often adds quotes or spaces)
        image_path = image_path.strip().strip('"')

        # Check if file exists *before* running
        if not os.path.exists(image_path):
            print(f"\n--- ERROR ---")
            print(f"File not found at path: {image_path}")
            print("Please make sure the path is correct and try again.")
        else:
            print(f"\nLoading image: {image_path}")
            classify_image(image_path)
    else:
        print("Invalid choice. Exiting.")