from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS
import os
import time
import sys

print("Python version:", sys.version)
print("Starting application...")
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir("."))

app = Flask(__name__)
CORS(app)

# Get absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load the model
try:
    from tensorflow.keras.models import model_from_json

    # Adjust the path for Render environment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # Check if model files exist and print paths for debugging
    model_json_path = os.path.join(MODEL_DIR, "emotion_model.json")
    model_weights_path = os.path.join(MODEL_DIR, "emotion_model.h5")

    print(f"Looking for model at: {model_json_path}")
    print(f"Looking for weights at: {model_weights_path}")

    if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
        print("ERROR: Model files not found at expected locations")

    json_file = open(model_json_path, "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback

    traceback.print_exc()

# Create a debug directory if it doesn't exist
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Load the face detector with multiple options
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Also load alternative face cascade for better detection
alt_haar_file = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
alt_face_cascade = cv2.CascadeClassifier(alt_haar_file)

if face_cascade.empty() or alt_face_cascade.empty():
    print(f"Error: One or more Haar cascade files not loaded properly!")
    exit(1)
else:
    print(f"Face detectors loaded successfully")

# Labels dictionary
LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}


def extract_feats(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255


@app.route("/predict", methods=["POST"])
def make_pred():
    try:
        # Generate a timestamp for debug files
        timestamp = int(time.time())

        file = request.files["image"].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Save the received image for debugging
        debug_input_path = os.path.join(DEBUG_DIR, f"input_{timestamp}.jpg")
        cv2.imwrite(debug_input_path, img)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray_eq = cv2.equalizeHist(gray)

        # Try multiple detection strategies with different parameters
        detection_strategies = [
            {
                "cascade": face_cascade,
                "scale": 1.3,
                "min_neighbors": 5,
                "name": "default",
            },
            {
                "cascade": face_cascade,
                "scale": 1.1,
                "min_neighbors": 3,
                "name": "sensitive",
            },
            {
                "cascade": alt_face_cascade,
                "scale": 1.1,
                "min_neighbors": 4,
                "name": "alt_default",
            },
            {
                "cascade": alt_face_cascade,
                "scale": 1.05,
                "min_neighbors": 3,
                "name": "alt_sensitive",
            },
        ]

        faces = []
        used_strategy = "none"

        # Try each strategy until we find faces
        for strategy in detection_strategies:
            detector = strategy["cascade"]
            scale = strategy["scale"]
            min_neighbors = strategy["min_neighbors"]

            # Try with regular grayscale
            faces = detector.detectMultiScale(gray, scale, min_neighbors)
            if len(faces) > 0:
                used_strategy = strategy["name"]
                break

            # If no faces found, try with equalized histogram
            faces = detector.detectMultiScale(gray_eq, scale, min_neighbors)
            if len(faces) > 0:
                used_strategy = strategy["name"] + "_equalized"
                break

        print(f"Detected faces: {len(faces)}, Strategy: {used_strategy}")

        if len(faces) == 0:
            # Save debug info
            debug_msg = f"No faces detected with any strategy"
            print(debug_msg)
            return jsonify({"error": "No face detected", "debug_info": debug_msg})

        # Mark faces on the debug image
        debug_img = img.copy()

        results = []
        for i, (p, q, r, s) in enumerate(faces):
            # Extract and process the face region
            face_region = gray[q : q + s, p : p + r]
            face_region_resized = cv2.resize(face_region, (48, 48))

            # Save the processed face
            face_debug_path = os.path.join(DEBUG_DIR, f"face_{timestamp}_{i}.jpg")
            cv2.imwrite(face_debug_path, face_region_resized)

            # Draw rectangle on debug image
            cv2.rectangle(debug_img, (p, q), (p + r, q + s), (0, 255, 0), 2)

            # Make prediction
            image_tensor = extract_feats(face_region_resized)
            prediction = model.predict(image_tensor)
            pred_idx = np.argmax(prediction)
            pred_lbl = LABELS[pred_idx]
            confidence = float(prediction[0][pred_idx])
            print(f"Prediction: {pred_lbl} (Confidence: {confidence:.2f})")

            # Add text to debug image
            cv2.putText(
                debug_img,
                f"{pred_lbl} ({confidence:.2f})",
                (p, q - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            results.append(
                {
                    "emotion": pred_lbl,
                    "confidence": float(confidence),
                    "position": {
                        "x": int(p),
                        "y": int(q),
                        "width": int(r),
                        "height": int(s),
                    },
                }
            )

        # Save debug image with annotations
        debug_output_path = os.path.join(DEBUG_DIR, f"output_{timestamp}.jpg")
        cv2.imwrite(debug_output_path, debug_img)

        return jsonify(
            {
                "results": results,
                "debug_info": {"strategy": used_strategy, "faces_found": len(faces)},
            }
        )

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from environment variable
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port)
