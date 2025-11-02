import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk  # <<< --- ADDED THIS IMPORT

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'office_goods_classifier.keras')
IMG_SIZE = (224, 224)
CLASSES = ['glass', 'keyboard', 'mouse', 'notebook', 'pen', 'stapler']

# --- Image Display Configuration ---
DISPLAY_MAX_WIDTH = 480
DISPLAY_MAX_HEIGHT = 854

# --- 1. Load Model ---
try:
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    root_check = tk.Tk()
    root_check.withdraw()
    messagebox.showerror(
        "Model Load Error",
        f"Error loading model: {e}\n\nPlease make sure '{MODEL_PATH}' is correct."
    )
    root_check.destroy()
    exit()


# --- 2. Core Classification Functions ---

def preprocess_image(frame, resize_shape):
    """Preprocesses a single frame for model prediction."""
    img = cv2.resize(frame, resize_shape)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    return img_batch


def post_process_prediction(prediction):
    """Takes the model's raw output and returns the class name and confidence."""
    pred_index = np.argmax(prediction[0])
    pred_class = CLASSES[pred_index]
    confidence = np.max(prediction[0]) * 100
    return pred_class, confidence


# --- MODIFIED THIS FUNCTION ---
def classify_image(image_path, root_window):
    """
    Classifies a single image file and displays it in a NEW TKINTER WINDOW.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        messagebox.showwarning("Image Error", f"Could not read image from {image_path}")
        return

    # --- Prediction ---
    processed_img = preprocess_image(frame.copy(), IMG_SIZE)
    prediction = model.predict(processed_img)
    pred_class, confidence = post_process_prediction(prediction)

    label_text = f"{pred_class} ({confidence:.1f}%)"
    print(f"Prediction: {label_text}")

    # --- Image Resizing for Display ---
    original_height, original_width = frame.shape[:2]
    aspect_ratio = original_width / original_height
    if original_width > DISPLAY_MAX_WIDTH or original_height > DISPLAY_MAX_HEIGHT:
        if (DISPLAY_MAX_WIDTH / DISPLAY_MAX_HEIGHT) > aspect_ratio:
            new_height = DISPLAY_MAX_HEIGHT
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = DISPLAY_MAX_WIDTH
            new_height = int(new_width / aspect_ratio)
    else:
        new_width = original_width
        new_height = original_height

    display_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Put text on the *resized* frame
    cv2.putText(display_frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- NEW: Convert OpenCV/Numpy image to Tkinter-compatible image ---
    # 1. Convert color from BGR (OpenCV) to RGB (PIL)
    img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    # 2. Convert Numpy array to PIL Image
    pil_image = Image.fromarray(img_rgb)
    # 3. Convert PIL Image to ImageTk.PhotoImage
    tk_image = ImageTk.PhotoImage(pil_image)

    # --- NEW: Create a new Toplevel window to display the image ---
    image_window = tk.Toplevel(root_window)
    image_window.title(f"Result: {pred_class}")
    image_window.resizable(False, False)

    # Center this pop-up over the main window
    image_window.geometry(f"+{root_window.winfo_x() + 50}+{root_window.winfo_y() + 50}")

    # Create a label to hold the image
    lbl_image = ttk.Label(image_window, image=tk_image)
    # IMPORTANT: Keep a reference to the image to prevent it from being garbage-collected
    lbl_image.image = tk_image
    lbl_image.pack(padx=10, pady=10)

    # Add a "Back" button to this window
    btn_back = ttk.Button(
        image_window,
        text="Back to Menu",
        command=image_window.destroy  # Just closes this pop-up
    )
    btn_back.pack(pady=10)

    # Make the window modal (disables main window until this one is closed)
    image_window.transient(root_window)
    image_window.grab_set()
    root_window.wait_window(image_window)  # Wait here until the user clicks "Back"


def classify_camera():
    """
    Classifies video from the default camera in real-time. (No changes needed)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        messagebox.showerror("Camera Error", "Could not open camera. Is it connected?")
        return

    print("Camera feed opened. Press 'q' to quit.")
    # (Rest of this function is unchanged)
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


# --- 3. GUI Helper Functions ---

# --- MODIFIED THIS FUNCTION ---
# (start_camera is unchanged)
def start_camera(root_window):
    """Hides the main menu and starts the live camera feed."""
    print("\nStarting live camera feed...")
    root_window.withdraw()  # Hide the main menu
    try:
        classify_camera()  # Run the existing classification function
    except Exception as f:
        print(f"An error occurred during camera classification: {f}")
        messagebox.showerror("Camera Error", f"An error occurred: {f}")
    finally:
        print("Camera feed closed. Returning to menu.")
        root_window.deiconify()  # Show the main menu again ("Back" button)


# --- MODIFIED THIS FUNCTION ---
def start_image_classification(root_window):
    """Opens a file dialog, then calls the function to show the image."""
    print("\nOpening file dialog...")
    image_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
    )

    if not image_path:  # User clicked 'Cancel'
        print("No file selected. Returning to menu.")
        return

    print(f"File selected: {image_path}")

    # --- CHANGED ---
    # We no longer hide/show the main window.
    # We just call classify_image and pass the root window to it,
    # so it can create a pop-up ON TOP of the main window.
    try:
        classify_image(image_path, root_window)  # Pass root_window
    except Exception as g:
        print(f"An error occurred during image classification: {g}")
        messagebox.showerror("Image Error", f"An error occurred: {g}")
    finally:
        print("Image window closed. Returning to menu.")


# --- 4. Main execution logic (Unchanged from previous GUI) ---
if __name__ == "__main__":
    print("Starting GUI...")

    root = tk.Tk()
    root.title("Office Goods Classifier")

    window_width = 400
    window_height = 220
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    root.resizable(False, False)

    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("TLabel", font=("Helvetica", 14, "bold"), padding=10)

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill=tk.BOTH)

    lbl_title = ttk.Label(main_frame, text="Select Classification Mode", anchor="center")
    lbl_title.pack(pady=(0, 20), fill=tk.X)

    btn_camera = ttk.Button(
        main_frame,
        text="Live Camera",
        command=lambda: start_camera(root)
    )
    btn_camera.pack(fill=tk.X, pady=5)

    btn_image = ttk.Button(
        main_frame,
        text="Classify Single Image File",
        command=lambda: start_image_classification(root)  # This command is the same
    )
    btn_image.pack(fill=tk.X, pady=5)

    root.mainloop()

    print("GUI closed. Exiting program.")