# Student Productivity Tracker

A real-time web application that detects student distraction (eye closure, yawning) using custom-trained CNNs and live webcam input. Built for the "Build Real ML Web Apps: No Wrappers, Just Real Models" hackathon.

## Features
- Live distraction detection: closed eyes, yawning
- Robust to normal blinks (blink smoothing)
- Real-time overlays and session stats
- All ML models trained from scratch (no wrappers, no APIs)
- Fully open source and reproducible

## Datasets Used
- **Eye State (Open/Closed):**
  - [MRL Eye Dataset (Kaggle)](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)
- **Yawn Detection:**
  - [Yawn Dataset (Kaggle)](https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset)

## Model Training
- See `backend/ml/train_eye_state.py` and `backend/ml/train_yawn.py` for full training code and instructions.
- All training scripts are open source and reproducible.

## Setup Instructions
1. **Clone the repo:**
   ```bash
   git clone <your-repo-url>
   cd <repo-root>
   ```
2. **Install backend dependencies:**
   ```bash
   pip install fastapi uvicorn torch torchvision pillow opencv-python
   ```
3. **Download datasets:**
   - MRL Eye Dataset: [Kaggle link](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)
   - Yawn Eye Dataset: [Kaggle link](https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset)
   - Place as:
     - `backend/ml/datasets/mrl/OpenEye/`, `backend/ml/datasets/mrl/ClosedEye/`
     - `backend/ml/datasets/yawn/yawn/`, `backend/ml/datasets/yawn/no_yawn/`
4. **Train models:**
   ```bash
   python -m backend.ml.train_eye_state
   python -m backend.ml.train_yawn
   # Move eye_cnn.pth and yawn_cnn.pth to backend/ml/
   ```
5. **Start backend server:**
   ```bash
   uvicorn backend.main:app --reload
   ```
6. **Serve frontend:**
   - Open `frontend/index.html` in your browser, or
   - Serve with a static server:
     ```bash
     cd frontend
     python -m http.server 8080
     ```



## References
- [MRL Eye Dataset (Kaggle)](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)
- [Yawn Eye Dataset (Kaggle)](https://www.kaggle.com/datasets/monu999/yawn-eye-dataset-new)
- [Mediapipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)

---

