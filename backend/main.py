from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
try:
    from backend.ml.train_eye_state import EyeCNN, get_transforms as get_eye_transforms
    from backend.ml.train_yawn import YawnCNN, get_transforms as get_yawn_transforms
except ModuleNotFoundError:
    from ml.train_eye_state import EyeCNN, get_transforms as get_eye_transforms
    from ml.train_yawn import YawnCNN, get_transforms as get_yawn_transforms
import time
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cpu')
eye_model = EyeCNN()
yawn_model = YawnCNN()

# Try both possible model paths for local and Render deployment
try:
    # For local development (run from project root)
    eye_model.load_state_dict(torch.load('backend/ml/eye_cnn.pth', map_location=device))
    yawn_model.load_state_dict(torch.load('backend/ml/yawn_cnn.pth', map_location=device))
except FileNotFoundError:
    # For Render (root is backend/)
    eye_model.load_state_dict(torch.load('ml/eye_cnn.pth', map_location=device))
    yawn_model.load_state_dict(torch.load('ml/yawn_cnn.pth', map_location=device))

eye_model.eval()
yawn_model.eval()

eye_transform = get_eye_transforms()
yawn_transform = get_yawn_transforms()

# Session state
session = {
    'last_status': 'Focused',
    'focused_time': 0.0,
    'distracted_time': 0.0,
    'last_update': time.time(),
    'closed_eye_frames': 0,  # For blink smoothing
    'distracted_threshold': 2  # Must be closed for 2+ frames (1s if polling every 0.5s)
}

def preprocess_upload(file: UploadFile, transform):
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = transform(img)
    img = img.unsqueeze(0)  # (1, 1, 32, 32)
    return img

@app.post("/api/frame")
async def receive_frame(
    left_eye: UploadFile = File(...),
    right_eye: UploadFile = File(...),
    mouth: UploadFile = File(...)
):
    # Preprocess and run inference for both eyes
    left_eye_img = preprocess_upload(left_eye, eye_transform)
    right_eye_img = preprocess_upload(right_eye, eye_transform)
    with torch.no_grad():
        left_pred = torch.argmax(eye_model(left_eye_img), dim=1).item()  # 0=open, 1=closed
        right_pred = torch.argmax(eye_model(right_eye_img), dim=1).item()
    eyes_closed = (left_pred == 1 or right_pred == 1)
    # Blink smoothing: count consecutive closed-eye frames
    if eyes_closed:
        session['closed_eye_frames'] += 1
    else:
        session['closed_eye_frames'] = 0
    # Preprocess and run inference for mouth
    mouth_img = preprocess_upload(mouth, yawn_transform)
    with torch.no_grad():
        yawn_pred = torch.argmax(yawn_model(mouth_img), dim=1).item()  # 0=no_yawn, 1=yawn
    # Distraction logic: only distracted if eyes closed for threshold frames or yawn
    if session['closed_eye_frames'] >= session['distracted_threshold'] or yawn_pred == 1:
        status = 'Distracted'
    else:
        status = 'Focused'
    # Time tracking
    now = time.time()
    elapsed = now - session['last_update']
    if session['last_status'] == 'Focused':
        session['focused_time'] += elapsed
    else:
        session['distracted_time'] += elapsed
    session['last_status'] = status
    session['last_update'] = now
    return JSONResponse({
        "status": status,
        "eyes_closed": eyes_closed,
        "yawn": bool(yawn_pred == 1),
        "focused_time": int(session['focused_time']),
        "distracted_time": int(session['distracted_time'])
    }) 