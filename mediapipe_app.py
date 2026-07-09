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

# --- Standard library imports ---
import os                  # used to check if the hand-tracking model file already exists on disk
import time                 # used for timestamps (FPS calculation, MediaPipe video timestamps)
import urllib.request       # used to download the hand-tracking model file from Google's servers

# --- Third-party library imports ---
import cv2                  # OpenCV: webcam capture, drawing shapes/text, image resizing, display window
import joblib                # used to load the trained ML model and label encoder saved as .pkl files
import mediapipe as mp       # Google's MediaPipe library: provides the hand-tracking model itself
import numpy as np           # numerical arrays: used for the canvas (image buffer) and math operations
from mediapipe.tasks.python import vision                       # MediaPipe Tasks API: hand landmark detection classes
from mediapipe.tasks.python.core.base_options import BaseOptions  # used to tell MediaPipe where the model file is
from skimage.feature import hog  # HOG (Histogram of Oriented Gradients): the feature extractor used during training,
                                   # must be applied the same way here so the model sees matching input

# ---------------------------------------------------------
# Downloading the hand-tracking model (one-time, first run only)
# ---------------------------------------------------------
# URL pointing to Google's hosted hand-tracking model file (the actual neural
# network weights that detect hand landmarks -- this is NOT your trained
# alphabet model, it's a separate general-purpose hand detector)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
# Local filename the model will be saved as, in the same folder as this script
MODEL_PATH = "hand_landmarker.task"

# Only download if it isn't already sitting on disk -- avoids re-downloading
# every time you run the script
if not os.path.exists(MODEL_PATH):
    print("Downloading hand-tracking model (one-time, ~8MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)  # actually fetches and saves the file
    print("Download complete.")

# ---------------------------------------------------------
# Loading the saved model files
# ---------------------------------------------------------
# These are YOUR trained artifacts from the alphabet recognition notebook --
# not related to MediaPipe. `model` is the trained classifier (e.g. SVM/RandomForest),
# `label_encoder` converts the model's numeric class output back into a letter (A-Z).
model = joblib.load("trained_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 640, 480  # resolution the webcam feed is captured/displayed at
CANVAS_SIZE = 400          # the virtual drawing canvas is CANVAS_SIZE x CANVAS_SIZE (in pixels)
TRAINING_IMAGE_SIZE = 34   # training images are 34x34, canvas is resized to this before prediction
LINE_THICKNESS = 8         # thickness (in pixels) of the line drawn on the canvas as you write
PINCH_THRESHOLD = 0.06     # normalized distance -- thumb+index closer than this = "pen down"
                           # (normalized means 0.0-1.0 relative to frame size, not pixels)

# --- Setting up the MediaPipe hand landmark detector ---
# BaseOptions tells MediaPipe which model file to load (the one we downloaded above)
base_options = BaseOptions(model_asset_path=MODEL_PATH)

# Configuration for the hand landmark detector:
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,                          # only track a single hand at a time (simpler, faster)
    min_hand_detection_confidence=0.7,    # how confident MediaPipe must be that a hand exists in frame
    min_hand_presence_confidence=0.6,     # confidence threshold to keep considering the hand "present"
    min_tracking_confidence=0.6,          # confidence threshold for tracking the hand across frames
    running_mode=vision.RunningMode.VIDEO,  # VIDEO mode expects a timestamp per frame (vs. single IMAGE mode)
)
# Actually create the detector object using the settings above -- this is what
# we'll call on every frame to get hand landmark positions
hand_landmarker = vision.HandLandmarker.create_from_options(hand_landmarker_options)


def extract_features(image_34x34):
    """Same HOG pipeline used during training (Step 5 of the notebook)."""
    # Normalize pixel values from 0-255 range down to 0-1 range (matches training preprocessing)
    img_array = image_34x34 / 255.0
    # Compute HOG features -- these describe edge/gradient patterns in the image,
    # which is what the trained classifier actually learned from, not raw pixels
    features = hog(
        img_array,
        orientations=9,              # number of gradient direction bins
        pixels_per_cell=(4, 4),      # size of each cell HOG groups pixels into
        cells_per_block=(2, 2),      # how many cells are grouped/normalized together
        block_norm="L2-Hys",         # the specific normalization method used
    )
    return features


def get_letter_bounding_box(canvas, padding=20):
    """Find the smallest box around what was actually drawn, with some padding."""
    # np.where finds all pixel coordinates where the canvas value is > 0 (i.e. part of the drawing)
    ys, xs = np.where(canvas > 0)
    # If nothing has been drawn yet, there's nothing to bound -- return None
    if len(xs) == 0 or len(ys) == 0:
        return None

    # Compute the tightest rectangle containing all drawn pixels, then add
    # padding on each side (clipped so it doesn't go outside canvas bounds)
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
    # Step 1: find the region of the canvas that actually contains the drawing
    bbox = get_letter_bounding_box(canvas)
    if bbox is None:
        # canvas is empty -- nothing to predict, so return "no result"
        return None, 0.0

    # Step 2: crop the canvas down to just that region
    x_min, x_max, y_min, y_max = bbox
    cropped = canvas[y_min:y_max, x_min:x_max]

    # Step 3: pad the crop to a square canvas so resizing doesn't distort the
    # letter's shape (e.g. stretching a tall thin "I" into a fat blob)
    h, w = cropped.shape
    side = max(h, w)                                   # size of the square = the larger dimension
    square = np.zeros((side, side), dtype=np.uint8)     # blank black square canvas
    y_offset = (side - h) // 2                          # center the crop vertically inside the square
    x_offset = (side - w) // 2                          # center the crop horizontally inside the square
    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped  # paste the cropped letter into the center

    # Step 4: resize the square down to exactly the training image size (34x34)
    resized = cv2.resize(square, (TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    # Step 5: extract HOG features the same way training data was processed,
    # then reshape into a single-row 2D array (model expects a batch of samples)
    features = extract_features(resized).reshape(1, -1)

    # Step 6: run the actual trained ML model on the extracted features
    prediction = model.predict(features)[0]              # predicted class (numeric label)
    probabilities = model.predict_proba(features)[0]     # confidence scores for every possible class

    # Step 7: convert numeric prediction back into an actual letter (e.g. 0 -> 'A')
    predicted_letter = label_encoder.inverse_transform([prediction])[0]
    # Step 8: confidence = highest probability among all classes, as a percentage
    confidence = round(max(probabilities) * 100, 2)

    return predicted_letter, confidence


def main():
    # --- Open the webcam ---
    cap = cv2.VideoCapture(0)                          # 0 = default system webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)         # request a specific capture width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)       # request a specific capture height

    # The virtual canvas the letter is actually drawn on (black background,
    # white stroke) -- this is what gets shown to the model, never the raw
    # camera frame.
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)  # blank black canvas to start

    prev_point = None          # last fingertip position drawn to (used to draw connecting lines)
    last_prediction = ""       # most recent predicted letter (shown on screen)
    last_confidence = 0.0      # most recent confidence score (shown on screen)

    prev_time = time.time()    # used to calculate FPS between frames
    start_time = time.time()   # reference point for MediaPipe's video timestamps

    print("Pinch your thumb and index finger together to draw a letter in the air.")
    print("Move without pinching to reposition. Press 'c' to clear, 'p' to predict, 'q' to quit.")

    # --- Main loop: runs once per webcam frame, forever, until 'q' is pressed ---
    while True:
        success, frame = cap.read()   # grab a single frame from the webcam
        if not success:
            break                     # webcam disconnected or failed -- stop the loop

        frame = cv2.flip(frame, 1)                          # mirror the image horizontally (feels natural, like a mirror)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV uses BGR color order, MediaPipe needs RGB

        # Wrap the frame for MediaPipe and run hand landmark detection.
        # detect_for_video needs a monotonically increasing timestamp (ms).
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  # wrap the numpy array in MediaPipe's image format
        timestamp_ms = int((time.time() - start_time) * 1000)                  # milliseconds since script started
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)      # run hand detection on this frame

        fingertip_point = None   # will hold the (x, y) canvas position of the index fingertip this frame
        is_pinching = False      # will be True if thumb and index finger are touching (pen down)

        # Only proceed if MediaPipe actually found a hand in this frame
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]  # first detected hand (we only track num_hands=1 anyway)

            # Landmark 8 is the index fingertip, landmark 4 is the thumb tip
            # (these landmark index numbers are fixed/standard in MediaPipe's hand model)
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            # Landmark coordinates come back normalized (0.0-1.0); convert to
            # actual pixel positions on the camera frame for drawing the dot
            x_px = int(index_tip.x * CAM_WIDTH)
            y_px = int(index_tip.y * CAM_HEIGHT)

            # Map the fingertip position from the camera frame onto the canvas
            # (canvas has its own separate resolution, CANVAS_SIZE x CANVAS_SIZE)
            canvas_x = int(index_tip.x * CANVAS_SIZE)
            canvas_y = int(index_tip.y * CANVAS_SIZE)
            fingertip_point = (canvas_x, canvas_y)

            # Pinch (thumb touching index finger) = "pen down". Moving the
            # hand around without pinching = "pen up", so repositioning
            # between strokes doesn't get drawn as a stray line.
            # Euclidean distance between thumb tip and index tip (in normalized coordinates)
            pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            is_pinching = pinch_distance < PINCH_THRESHOLD   # close enough = considered a pinch

            # Draw a dot on the live camera feed at the fingertip position:
            # green while pinching (drawing), red while not (just moving)
            dot_color = (0, 255, 0) if is_pinching else (0, 0, 255)
            cv2.circle(frame, (x_px, y_px), 8, dot_color, -1)

        # Draw the trail on the canvas -- only while pinching
        if fingertip_point is not None and is_pinching:
            if prev_point is not None:
                # draw a white line from the previous fingertip position to the current one
                # (drawing single points wouldn't produce a continuous stroke -- connecting
                # consecutive points frame-to-frame is what creates a smooth line)
                cv2.line(canvas, prev_point, fingertip_point, 255, LINE_THICKNESS)
            prev_point = fingertip_point   # remember this position for the next frame's line segment
        else:
            # pen is "up" (not pinching, or no hand detected) -- don't connect
            # the next stroke to wherever the finger was last, so repositioning
            # doesn't create an unwanted line
            prev_point = None

        # --- FPS counter (just for on-screen debug info, not used for logic) ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        # --- Build the display frame: camera feed + small canvas preview + text ---
        canvas_preview = cv2.resize(canvas, (200, 200))                    # shrink canvas for the on-screen preview box
        canvas_preview_bgr = cv2.cvtColor(canvas_preview, cv2.COLOR_GRAY2BGR)  # convert grayscale canvas to BGR so it can be pasted into the color camera frame

        # Draw a cursor on the preview showing exactly where the fingertip
        # currently maps to on the canvas -- even while the pen is "up" (not
        # pinching). This makes it much easier to line up separate strokes,
        # like the three bars of a capital E, with what's already drawn.
        if fingertip_point is not None:
            # scale the fingertip's canvas-space position down to the smaller 200x200 preview size
            preview_x = int(np.clip(fingertip_point[0] * 200 / CANVAS_SIZE, 0, 199))
            preview_y = int(np.clip(fingertip_point[1] * 200 / CANVAS_SIZE, 0, 199))
            cursor_color = (0, 255, 0) if is_pinching else (0, 0, 255)   # green = drawing, red = repositioning
            cv2.drawMarker(canvas_preview_bgr, (preview_x, preview_y), cursor_color,
                            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # Paste the small canvas preview into the top-right corner of the camera frame
        frame[10:210, CAM_WIDTH - 210:CAM_WIDTH - 10] = canvas_preview_bgr
        # Draw a white border rectangle around the preview box so it's visually separated from the camera feed
        cv2.rectangle(frame, (CAM_WIDTH - 210, 10), (CAM_WIDTH - 10, 210), (255, 255, 255), 2)

        # On-screen text: FPS counter (top-left) and control instructions (bottom)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "pinch: draw   c: clear   p: predict   q: quit", (10, CAM_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If a prediction has been made, display it near the top of the frame
        if last_prediction:
            cv2.putText(frame, f"Prediction: {last_prediction}  ({last_confidence}%)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        # Actually show the assembled frame in a window
        cv2.imshow("Alphabet Air-Writing Recognition", frame)

        # Wait 1ms for a keypress, and read which key (if any) was pressed
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break   # quit the app
        elif key == ord("c"):
            # clear everything: reset the canvas to blank, forget the last stroke
            # position, and clear any previous prediction shown on screen
            canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
            prev_point = None
            last_prediction = ""
            last_confidence = 0.0
        elif key == ord("p"):
            # run prediction on whatever has been drawn on the canvas so far
            predicted_letter, confidence = predict_letter(canvas)
            if predicted_letter is None:
                print("Canvas is empty -- draw a letter first.")
            else:
                last_prediction, last_confidence = predicted_letter, confidence
                print(f"Predicted letter: {last_prediction}  (confidence: {last_confidence}%)")

    # --- Cleanup after the loop ends (webcam closed or 'q' pressed) ---
    cap.release()             # release the webcam device so other apps can use it
    cv2.destroyAllWindows()   # close any OpenCV display windows


if __name__ == "__main__":
    main()   # only run main() if this file is executed directly (not imported as a module)
