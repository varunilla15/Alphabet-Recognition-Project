"""
Handwritten Alphabet Recognition - Live Air-Writing App
Innomatics Research Labs - Image Classification Capstone Project

Draw a letter in the air with your index fingertip. The app tracks your
fingertip and draws the trail on a virtual black canvas (not the raw camera
image) -- this keeps the drawn image in the same style as the training data
(a white stroke on a black background), which is what actually makes live
recognition work.

NOTE ON THE MEDIAPIPE API: this uses the newer "MediaPipe Tasks" API
(mp.tasks.vision.HandLandmarker), not the older mp.solutions.hands API seen
in a lot of tutorials online. Google removed mp.solutions in recent
mediapipe releases (0.10.31+), so mp.solutions.hands no longer exists in
current installs -- this file uses the API that replaced it.

On first run, this script automatically downloads a small hand-tracking
model file (hand_landmarker.task, ~8MB) from Google's servers and saves it
next to this script. You need an internet connection the first time only.

Controls:
    pinch (thumb + index finger together) - draw
    move without pinching                 - reposition without drawing
    c  - clear the canvas
    p  - predict the letter you drew
    q  - quit

Run with:
    python mediapipe_app.py
"""

import os
import time
import urllib.request

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from skimage.feature import hog

# ---------------------------------------------------------
# Downloading the hand-tracking model (one-time, first run only)
# ---------------------------------------------------------
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand-tracking model (one-time, ~8MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ---------------------------------------------------------
# Loading the saved model files
# ---------------------------------------------------------
model = joblib.load("trained_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 640, 480
CANVAS_SIZE = 400          # the virtual drawing canvas is CANVAS_SIZE x CANVAS_SIZE
TRAINING_IMAGE_SIZE = 34   # training images are 34x34, canvas is resized to this before prediction
LINE_THICKNESS = 8
PINCH_THRESHOLD = 0.06     # normalized distance -- thumb+index closer than this = "pen down"

base_options = BaseOptions(model_asset_path=MODEL_PATH)
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=vision.RunningMode.VIDEO,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_landmarker_options)


def extract_features(image_34x34):
    """Same HOG pipeline used during training (Step 5 of the notebook)."""
    img_array = image_34x34 / 255.0
    features = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def get_letter_bounding_box(canvas, padding=20):
    """Find the smallest box around what was actually drawn, with some padding."""
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = max(0, xs.min() - padding)
    x_max = min(canvas.shape[1], xs.max() + padding)
    y_min = max(0, ys.min() - padding)
    y_max = min(canvas.shape[0], ys.max() + padding)
    return x_min, x_max, y_min, y_max


def predict_letter(canvas):
    """Crop to the drawn letter, center it in a square, resize to 34x34, and predict.

    Training images are tightly cropped around the letter (it fills almost
    the whole frame). Simply resizing the full, mostly-empty canvas would
    leave the letter tiny and off-center relative to what the model learned,
    so we crop to the drawn strokes first and pad to a square to avoid
    stretching the letter out of shape.
    """
    bbox = get_letter_bounding_box(canvas)
    if bbox is None:
        return None, 0.0

    x_min, x_max, y_min, y_max = bbox
    cropped = canvas[y_min:y_max, x_min:x_max]

    # Pad the crop to a square so resizing doesn't distort the letter's shape
    h, w = cropped.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    resized = cv2.resize(square, (TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    features = extract_features(resized).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_letter = label_encoder.inverse_transform([prediction])[0]
    confidence = round(max(probabilities) * 100, 2)

    return predicted_letter, confidence


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # The virtual canvas the letter is actually drawn on (black background,
    # white stroke) -- this is what gets shown to the model, never the raw
    # camera frame.
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    prev_point = None
    last_prediction = ""
    last_confidence = 0.0

    prev_time = time.time()
    start_time = time.time()

    print("Pinch your thumb and index finger together to draw a letter in the air.")
    print("Move without pinching to reposition. Press 'c' to clear, 'p' to predict, 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the frame for MediaPipe and run hand landmark detection.
        # detect_for_video needs a monotonically increasing timestamp (ms).
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        fingertip_point = None
        is_pinching = False

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]  # first detected hand

            # Landmark 8 is the index fingertip, landmark 4 is the thumb tip
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            x_px = int(index_tip.x * CAM_WIDTH)
            y_px = int(index_tip.y * CAM_HEIGHT)

            # Map the fingertip position from the camera frame onto the canvas
            canvas_x = int(index_tip.x * CANVAS_SIZE)
            canvas_y = int(index_tip.y * CANVAS_SIZE)
            fingertip_point = (canvas_x, canvas_y)

            # Pinch (thumb touching index finger) = "pen down". Moving the
            # hand around without pinching = "pen up", so repositioning
            # between strokes doesn't get drawn as a stray line.
            pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            is_pinching = pinch_distance < PINCH_THRESHOLD

            dot_color = (0, 255, 0) if is_pinching else (0, 0, 255)
            cv2.circle(frame, (x_px, y_px), 8, dot_color, -1)

        # Draw the trail on the canvas -- only while pinching
        if fingertip_point is not None and is_pinching:
            if prev_point is not None:
                cv2.line(canvas, prev_point, fingertip_point, 255, LINE_THICKNESS)
            prev_point = fingertip_point
        else:
            prev_point = None

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        # Build the display frame: camera feed + small canvas preview + text
        canvas_preview = cv2.resize(canvas, (200, 200))
        canvas_preview_bgr = cv2.cvtColor(canvas_preview, cv2.COLOR_GRAY2BGR)

        # Draw a cursor on the preview showing exactly where the fingertip
        # currently maps to on the canvas -- even while the pen is "up" (not
        # pinching). This makes it much easier to line up separate strokes,
        # like the three bars of a capital E, with what's already drawn.
        if fingertip_point is not None:
            preview_x = int(np.clip(fingertip_point[0] * 200 / CANVAS_SIZE, 0, 199))
            preview_y = int(np.clip(fingertip_point[1] * 200 / CANVAS_SIZE, 0, 199))
            cursor_color = (0, 255, 0) if is_pinching else (0, 0, 255)
            cv2.drawMarker(canvas_preview_bgr, (preview_x, preview_y), cursor_color,
                            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        frame[10:210, CAM_WIDTH - 210:CAM_WIDTH - 10] = canvas_preview_bgr
        cv2.rectangle(frame, (CAM_WIDTH - 210, 10), (CAM_WIDTH - 10, 210), (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "pinch: draw   c: clear   p: predict   q: quit", (10, CAM_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if last_prediction:
            cv2.putText(frame, f"Prediction: {last_prediction}  ({last_confidence}%)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        cv2.imshow("Alphabet Air-Writing Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
            prev_point = None
            last_prediction = ""
            last_confidence = 0.0
        elif key == ord("p"):
            predicted_letter, confidence = predict_letter(canvas)
            if predicted_letter is None:
                print("Canvas is empty -- draw a letter first.")
            else:
                last_prediction, last_confidence = predicted_letter, confidence
                print(f"Predicted letter: {last_prediction}  (confidence: {last_confidence}%)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
