import cv2
import torch
import numpy as np
import sqlite3
import threading
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
from queue import Queue
import telegram
from threading import Lock

# -------------------------------
# Configuration
# -------------------------------
CONFIDENCE_THRESHOLD = 0.65  # Adjust between 0.5 - 0.6
LINE_THRESHOLD = 20  # Distance (pixels) to consider "on" a line

# User Input: Choose Video File or Webcam
mode = input("Enter '1' for webcam or '2' for video file: ")
if mode == '2':
    VIDEO_FILE = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(VIDEO_FILE)
    # cap = cv2.VideoCapture("Punching.mp4")
else:
    cap = cv2.VideoCapture(0)  # Default webcam

# -------------------------------
# Helper functions
# -------------------------------

def point_line_distance(point, line):
    """
    Calculate the perpendicular distance from a point to a line.
    :param point: Tuple (x0, y0)
    :param line: Tuple of two points ((x1, y1), (x2, y2))
    :return: Distance (float)
    """
    (x0, y0) = point
    ((x1, y1), (x2, y2)) = line
    numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    denominator = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
    return numerator / denominator if denominator != 0 else float('inf')

def select_line(window_name, frame):
    """
    Let the user select a line by clicking two points on the frame.
    Press 'c' to confirm the selection.
    :param window_name: Name of the display window.
    :param frame: Image frame (numpy array) on which to select the line.
    :return: Tuple of two points ((x1, y1), (x2, y2))
    """
    points = []
    clone = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal clone
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, clone)

    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_callback)
    print(f"Select two points for {window_name} and press 'c' to confirm.")

    # Wait until the user presses 'c' (confirm) after selecting at least two points
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(points) >= 2:
            break
    cv2.setMouseCallback(window_name, lambda *args: None)  # disable callback
    return points[0], points[1]

# -------------------------------
# Main initialization and setup
# -------------------------------

# Enable GPU acceleration if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# Load YOLOv8 model
print("[INFO] Loading YOLOv8 model...")
model = YOLO("yolov8m.pt").to(device)
print("[INFO] YOLOv8 model loaded successfully.")

# Initialize DeepSORT tracker (Real-time version)
print("[INFO] Initializing DeepSORT tracker...")
tracker = DeepSort(max_age=70, n_init=3, nn_budget=100)

# Telegram bot setup
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
print("[INFO] Telegram bot initialized.")

# SQLite database connection
conn = sqlite3.connect("violations.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS Violations (UserID TEXT, Time TEXT, Issue TEXT)")
db_lock = Lock()
print("[INFO] Database connected and table ensured.")

# -------------------------------
# Select Punching & Crossing Lines
# -------------------------------
ret, init_frame = cap.read()
if not ret:
    print("[ERROR] Could not read first frame for line selection.")
    exit()

punching_line = select_line("Select Punching Line", init_frame.copy())
print(f"[INFO] Punching line selected: {punching_line}")

crossing_line = select_line("Select Crossing Line", init_frame.copy())
print(f"[INFO] Crossing line selected: {crossing_line}")

# Close the selection windows
cv2.destroyWindow("Select Punching Line")
cv2.destroyWindow("Select Crossing Line")

# -------------------------------
# Tracking & Violation Storage
# -------------------------------
# Dictionaries for tracking user states and recording violations
user_tracking = {}           # Format: { track_id: {"punched": datetime, "crossed": bool} }
violations_recorded = {}     # Format: { track_id: date }
violation_queue = Queue(maxsize=100)

# MQTT setup for real-time alerting (currently commented out)
# mqtt_client = mqtt.Client()
# mqtt_client.connect("mqtt_broker_address", 1883, 60)

# -------------------------------
# Violation processing thread
# -------------------------------

def process_violations():
    while True:
        track_id, punch_time = violation_queue.get()
        alert_msg = f"⚠️ Alert: User {track_id} punched but didn't cross!"
        print(f"[VIOLATION] Processing violation: {alert_msg}")

        with db_lock:
            cursor.execute("INSERT INTO Violations VALUES (?, ?, ?)", (track_id, punch_time, "Did not cross"))
            conn.commit()

        # mqtt_client.publish("alerts/violations", alert_msg)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=alert_msg)
        print(f"[ALERT] Telegram alert sent for User {track_id}")
        # violation_queue.task_done()

threading.Thread(target=process_violations, daemon=True).start()
print("[INFO] Violation processing thread started.")

# -------------------------------
# Main processing loop
# -------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    # Draw the selected punching and crossing lines on the frame
    cv2.line(frame, punching_line[0], punching_line[1], (0, 0, 255), 2)  # Red line for punching
    cv2.line(frame, crossing_line[0], crossing_line[1], (0, 255, 0), 2)  # Green line for crossing

    # Run YOLOv8 detection
    results = model(frame, verbose=False)
    detections = []
    print("[INFO] Processing detections...")

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = box  # Bounding box coordinates
            if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD:  # Assuming class 0 is "person" and check for "person" and confidence score
                # Draw detection bounding box (blue) and record detection
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 20, 147), 2)
                detections.append([[x1, y1, x2, y2], conf])
                print(f"[DETECTION] Person detected at: {(x1, y1, x2, y2)} with confidence {conf:.2f}")

    # Update tracks using DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)
    print(f"[INFO] Number of active tracks: {len(tracks)}")

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_tlbr()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Default ID Color: Magenta
        id_color = (255, 0, 255)

        # Check if the center is near the punching line
        if point_line_distance((center_x, center_y), punching_line) < LINE_THRESHOLD:
            print("Center is near the punching line")
            id_color = 255, 165, 0  # Change to Orange

        # Check if the center is near the crossing line
        if point_line_distance((center_x, center_y), crossing_line) < LINE_THRESHOLD:
            print("Center is near the crossing line")
            id_color = (0, 255, 127)  # Change to Spring Green

        # Draw the tracked bounding box (green) with the assigned ID
        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)

        print(f"[TRACK] Tracking ID: {track_id}, Center: ({center_x:.1f}, {center_y:.1f})")

        # Check if the center is near the punching line.
        distance_to_punch = point_line_distance((center_x, center_y), punching_line)
        if distance_to_punch < LINE_THRESHOLD:
            if track_id not in user_tracking:
                user_tracking[track_id] = {"punched": datetime.now(), "crossed": False}
                print(f"[PUNCH] User {track_id} punched at {user_tracking[track_id]['punched']} "
                      f"(distance: {distance_to_punch:.2f})")

        # Check if the center is near the crossing line.
        distance_to_cross = point_line_distance((center_x, center_y), crossing_line)
        if distance_to_cross < LINE_THRESHOLD:
            if track_id in user_tracking and not user_tracking[track_id]["crossed"]:
                user_tracking[track_id]["crossed"] = True
                print(f"[CROSS] User {track_id} successfully crossed "
                      f"(distance: {distance_to_cross:.2f})")

    # Check for users who have been punched but not crossed after a delay
    for track_id, data in list(user_tracking.items()):
        if data["punched"] and not data["crossed"]:
            elapsed_time = (datetime.now() - data["punched"]).seconds
            if elapsed_time > 60:
                current_date = datetime.now().date()
                if track_id in violations_recorded and violations_recorded[track_id] == current_date:
                    print(f"[INFO] User {track_id} already recorded for today, skipping.")
                    continue
                print(f"[VIOLATION] Adding User {track_id} to violation queue.")
                violation_queue.put((track_id, data["punched"]))
                violations_recorded[track_id] = current_date

    print(f"[QUEUE] Current Violation Queue Size: {violation_queue.qsize()}")
    cv2.imshow("Live CCTV Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
conn.close()
print("[INFO] Cleanup completed.")