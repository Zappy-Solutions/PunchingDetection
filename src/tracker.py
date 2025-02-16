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
# max_age=70: The maximum number of frames a track can be inactive before it's deleted.
# n_init=3: The minimum number of frames a track must be visible to be confirmed.
# nn_budget=100: The maximum number of nearest neighbors to consider when matching tracks.
tracker = DeepSort(max_age=10, n_init=3, nn_budget=75)
logging.info("[INFO] DeepSORT tracker initialized.")
