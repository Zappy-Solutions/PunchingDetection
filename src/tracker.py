# tracker.py
import torch
import logging
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"[INFO] Using device: {device}")

# Load YOLOv8 model
model = YOLO("yolov8m.pt").to(device)
logging.info("[INFO] YOLOv8 model loaded successfully.")

# Initialize DeepSORT tracker
# Here we set max_age to 150 so that tracks persist longer.
# n_init=3: minimum number of frames a track must be visible to be confirmed.
# nn_budget=100: maximum number of nearest neighbors considered.
tracker = DeepSort(max_age=300, n_init=5, nn_budget=100)
logging.info("[INFO] DeepSORT tracker initialized.")
