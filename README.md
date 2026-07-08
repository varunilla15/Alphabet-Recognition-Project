# Handwritten Alphabet Recognition using Machine Learning

An Image Classification Capstone Project — Innomatics Research Labs

## Project Overview

This project builds a machine learning model that recognizes a single handwritten letter (A-Z) from an image, and connects it to a **live, camera-based air-writing app** — draw a letter in the air with your fingertip, and the model predicts which letter you drew.

## Problem Statement

Given an image of a single handwritten letter, predict which of the 26 English alphabet letters (A-Z) it is.

**Input:** `34x34` grayscale image — a white letter stroke on a black background.
**Target Variable:** `letter` (26 classes, A-Z)

## Dataset

- **Folder:** `dataset/` — one subfolder per letter (`A/`, `B/`, ... `Z/`)
- **Total Images:** 6,831
- All images are `34x34` grayscale, no missing or corrupted files
- Class sizes vary somewhat (from ~200 to ~390 images per letter) but every class has enough samples to train on

## Why an Air-Writing Approach

The dataset images are **drawn strokes** (white line on black background), not photographs of hands or objects. Feeding a real webcam photo directly into this model would fail, because a photo's pixel statistics (skin tone, lighting, texture) look nothing like a clean drawn stroke.

Instead, the camera app tracks your index fingertip using MediaPipe and **draws the trail itself** on a virtual black canvas — the model only ever sees this synthetic drawn canvas, never the raw camera frame. This keeps the live input in the same visual style as the training data, which is what actually makes the recognition work.

## Workflow

The pipeline (see `notebook.ipynb`) follows these steps:

1. Problem Statement
2. Importing the Dataset
3. Exploratory Data Analysis (EDA)
4. Image Preprocessing (normalization)
5. Feature Extraction (HOG)
6. Input-Output Separation
7. Train-Test Split (80/20)
8. Model Building (4 baseline algorithms compared)
9. Hyperparameter Tuning (GridSearchCV on the best model)
10. Model Evaluation
11. Model Saving
12. Model Loading
13. Camera Integration (`mediapipe_app.py`)
14. User Interface (`streamlit_app.py`)
15. Final Conclusion

## Algorithms

| Model | Accuracy |
|---|---|
| **KNN (tuned)** | **0.9217** |
| Logistic Regression | ~0.916 |
| Random Forest | ~0.890 |
| Decision Tree | ~0.605 |

**KNN** was selected as the final model. It gave the best accuracy of all baseline models, and since this app only predicts once per button-press (not once per video frame), KNN's comparison-based prediction cost is not a practical concern here — unlike a per-frame use case, where a faster model would be worth an accuracy tradeoff.

## Results

**Best Model:** K-Nearest Neighbors (tuned)
**Best Parameters:** `n_neighbors=3, weights='distance'`

| Metric | Score |
|---|---|
| Accuracy | 0.9217 |
| Precision | 0.9229 |
| Recall | 0.9217 |
| F1 Score | 0.9215 |

## Installation

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. No NLTK/external downloads are needed for this project.

## Running the Notebook

Open `notebook.ipynb` in Jupyter Notebook or JupyterLab and run all cells in order.

## Running the Air-Writing App

Make sure `trained_model.pkl` and `label_encoder.pkl` are in the same folder as `mediapipe_app.py`, then run:

```
python mediapipe_app.py
```

Hold your hand up to the camera and draw a letter in the air with your index fingertip.
- `c` — clear the canvas
- `p` — predict the letter you drew
- `q` — quit

## Running the Streamlit App

```
streamlit run streamlit_app.py
```

Upload a letter image (ideally a white stroke on black background) and get a prediction with confidence score.

## Folder Structure

```
Alphabet_Recognition_Project/
│
├── notebook.ipynb          # Full ML pipeline, step by step
├── mediapipe_app.py         # Live camera air-writing recognition app
├── streamlit_app.py         # Web UI for uploading a letter image
├── requirements.txt         # Python dependencies
├── README.md                 # Project documentation (this file)
├── trained_model.pkl         # Saved KNN model
├── label_encoder.pkl         # Saved label encoder
├── dataset/                  # Dataset (A/ ... Z/ subfolders of .jpg images)
├── assets/                   # Saved plots (EDA, evaluation charts)
└── models/                   # Duplicate copies of the saved model files
```

## Limitations

- The training data is drawn strokes, not photographs, so any live input must be *recreated* in that same style (via the air-writing canvas) rather than photographed directly.
- Only single, static letters are supported — no cursive writing or full-word recognition.

## Future Improvements

- Collect real air-written samples (captured through the app itself) to fine-tune the model on data closer to live usage.
- Try a CNN trained directly on raw pixel images and compare against the HOG + KNN pipeline.
- Extend to full-word recognition, one letter at a time.
- Deploy the Streamlit app publicly for easy demoing.

## References

- Scikit-Learn Documentation: https://scikit-learn.org/stable/
- Scikit-Image HOG Documentation: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
- MediaPipe Hands Documentation: https://developers.google.com/mediapipe
