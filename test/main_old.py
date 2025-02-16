# # import cv2
# # import torch
# # import numpy as np
# # import sqlite3
# # import threading
# # # import ffmpeg
# # import paho.mqtt.client as mqtt
# # from ultralytics import YOLO
# # # from bytetrack import BYTETracker
# # from yolox.tracker.byte_tracker import BYTETracker
# # from deep_sort.deep_sort.deep_sort import DeepSort
# # from datetime import datetime
# # from queue import Queue
# # import telegram
# #
# #
# # # Enable GPU acceleration
# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #
# # # Load YOLOv8 TensorRT optimized model
# # model = YOLO("yolov8m.engine").to(device)
# #
# # # Initialize ByteTrack & DeepSORT
# # tracker = BYTETracker()
# # deep_sort = DeepSort("ckpt.t7")
# #
# # # Telegram bot setup
# # TELEGRAM_BOT_TOKEN = "your_bot_token"
# # TELEGRAM_CHAT_ID = "your_chat_id"
# # bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
# #
# # # SQLite database connection
# # conn = sqlite3.connect("violations.db", check_same_thread=False)
# # cursor = conn.cursor()
# # cursor.execute("CREATE TABLE IF NOT EXISTS Violations (UserID TEXT, Time TEXT, Issue TEXT)")
# #
# # # Define regions
# # PUNCHING_ZONE = (50, 50, 200, 150)
# # CROSSING_ZONE = (300, 300, 600, 400)
# #
# # user_tracking = {}
# # violation_queue = Queue()
# #
# # # MQTT setup for real-time alerting
# # mqtt_client = mqtt.Client()
# # mqtt_client.connect("mqtt_broker_address", 1883, 60)
# #
# # # Process violations asynchronously
# # def process_violations():
# #     while True:
# #         track_id, punch_time = violation_queue.get()
# #         alert_msg = f"⚠️ Alert: User {track_id} punched but didn't cross!"
# #         cursor.execute("INSERT INTO Violations VALUES (?, ?, ?)", (track_id, punch_time, "Did not cross"))
# #         conn.commit()
# #         mqtt_client.publish("alerts/violations", alert_msg)
# #         bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=alert_msg)
# #         violation_queue.task_done()
# #
# # threading.Thread(target=process_violations, daemon=True).start()
# #
# # # Webcam (Windows compatible)
# # cap = cv2.VideoCapture(0)  # Use webcam (0 for default camera)
# #
# # # ToDo: GStreamer for optimized RTSP streaming
# # # cap = cv2.VideoCapture("rtsp://your_camera_ip", cv2.CAP_GSTREAMER)
# #
# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     frame = cv2.resize(frame, (1280, 720))
# #     results = model(frame, verbose=False)
# #     detections = []
# #
# #     for result in results.xyxy[0]:
# #         x1, y1, x2, y2, conf, cls = result
# #         if int(cls) == 0:
# #             detections.append([x1, y1, x2, y2, conf])
# #
# #     tracks = tracker.update(np.array(detections), frame)
# #     deep_sort.update(tracks, frame)
# #
# #     for track in tracks:
# #         x1, y1, x2, y2, track_id = track
# #         center_x = (x1 + x2) / 2
# #         center_y = (y1 + y2) / 2
# #
# #         if PUNCHING_ZONE[0] < center_x < PUNCHING_ZONE[2] and PUNCHING_ZONE[1] < center_y < PUNCHING_ZONE[3]:
# #             if track_id not in user_tracking:
# #                 user_tracking[track_id] = {"punched": datetime.now(), "crossed": False}
# #
# #         if CROSSING_ZONE[0] < center_x < CROSSING_ZONE[2] and CROSSING_ZONE[1] < center_y < CROSSING_ZONE[3]:
# #             if track_id in user_tracking:
# #                 user_tracking[track_id]["crossed"] = True
# #
# #     for track_id, data in list(user_tracking.items()):
# #         if data["punched"] and not data["crossed"]:
# #             elapsed_time = (datetime.now() - data["punched"]).seconds
# #             if elapsed_time > 15:
# #                 violation_queue.put((track_id, data["punched"]))
# #
# #     cv2.imshow("Live CCTV Monitoring", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
# # conn.close()
#
#
#
# #################### NEW CODE #####################
# import cv2
# import torch
# import numpy as np
# import sqlite3
# import threading
# import paho.mqtt.client as mqtt
# from ultralytics import YOLO
# from yolox.tracker.byte_tracker import BYTETracker
# from deep_sort.deep_sort.deep_sort import DeepSort
# from datetime import datetime
# from queue import Queue
# import telegram
# from threading import Lock
#
# # Enable GPU acceleration
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # Load YOLOv8 model
# model = YOLO("yolov8m.pt").to(device)
#
# # Initialize ByteTrack & DeepSORT
# tracker = BYTETracker()
# deep_sort = DeepSort(
#     model_path="ckpt.t7",
#     max_dist=0.2,
#     min_confidence=0.3,
#     nms_max_overlap=1.0,
#     max_iou_distance=0.7,
#     max_age=70,
#     n_init=3,
#     nn_budget=100,
#     use_cuda=True
# )
#
# # Telegram bot setup
# TELEGRAM_BOT_TOKEN = "your_bot_token"
# TELEGRAM_CHAT_ID = "your_chat_id"
# bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
#
# # SQLite database connection
# conn = sqlite3.connect("violations.db", check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute("CREATE TABLE IF NOT EXISTS Violations (UserID TEXT, Time TEXT, Issue TEXT)")
# db_lock = Lock()
#
# # Define regions
# PUNCHING_ZONE = (50, 50, 200, 150)
# CROSSING_ZONE = (300, 300, 600, 400)
#
# user_tracking = {}
# violation_queue = Queue(maxsize=100)
#
# # MQTT setup for real-time alerting
# mqtt_client = mqtt.Client()
# mqtt_client.connect("mqtt_broker_address", 1883, 60)
#
# # Process violations asynchronously
# def process_violations():
#     while True:
#         track_id, punch_time = violation_queue.get()
#         alert_msg = f"⚠️ Alert: User {track_id} punched but didn't cross!"
#         with db_lock:
#             cursor.execute("INSERT INTO Violations VALUES (?, ?, ?)", (track_id, punch_time, "Did not cross"))
#             conn.commit()
#         mqtt_client.publish("alerts/violations", alert_msg)
#         bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=alert_msg)
#         violation_queue.task_done()
#
# threading.Thread(target=process_violations, daemon=True).start()
#
# # Webcam (Windows compatible)
# cap = cv2.VideoCapture(0)  # Use webcam (0 for default camera)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame = cv2.resize(frame, (1280, 720))
#     results = model(frame, verbose=False)
#     detections = []
#
#     for result in results.xyxy[0]:
#         x1, y1, x2, y2, conf, cls = result
#         if int(cls) == 0:  # Assuming class 0 is "person"
#             detections.append([x1, y1, x2 - x1, y2 - y1, conf])  # Format: [x1, y1, w, h, conf]
#
#     tracks = tracker.update(np.array(detections), frame)
#     deep_sort_tracks = []
#     for track in tracks:
#         x1, y1, x2, y2, track_id = track
#         deep_sort_tracks.append([x1, y1, x2 - x1, y2 - y1, track_id])
#     deep_sort.update(deep_sort_tracks, frame)
#
#     for track in tracks:
#         x1, y1, x2, y2, track_id = track
#         center_x = (x1 + x2) / 2
#         center_y = (y1 + y2) / 2
#
#         if PUNCHING_ZONE[0] < center_x < PUNCHING_ZONE[2] and PUNCHING_ZONE[1] < center_y < PUNCHING_ZONE[3]:
#             if track_id not in user_tracking:
#                 user_tracking[track_id] = {"punched": datetime.now(), "crossed": False}
#
#         if CROSSING_ZONE[0] < center_x < CROSSING_ZONE[2] and CROSSING_ZONE[1] < center_y < CROSSING_ZONE[3]:
#             if track_id in user_tracking:
#                 user_tracking[track_id]["crossed"] = True
#
#     for track_id, data in list(user_tracking.items()):
#         if data["punched"] and not data["crossed"]:
#             elapsed_time = (datetime.now() - data["punched"]).seconds
#             if elapsed_time > 15:
#                 violation_queue.put((track_id, data["punched"]))
#
#     cv2.imshow("Live CCTV Monitoring", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# conn.close()



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

# Enable GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model
model = YOLO("yolov8m.pt").to(device)

# Initialize DeepSORT (Real-time version)
tracker = DeepSort(max_age=70, n_init=3, nn_budget=100)

# Telegram bot setup
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# SQLite database connection
conn = sqlite3.connect("violations.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS Violations (UserID TEXT, Time TEXT, Issue TEXT)")
db_lock = Lock()

# Define regions
PUNCHING_ZONE = (50, 50, 200, 150)
CROSSING_ZONE = (300, 300, 600, 400)

user_tracking = {}
violation_queue = Queue(maxsize=100)

# MQTT setup for real-time alerting
# mqtt_client = mqtt.Client()
# mqtt_client.connect("mqtt_broker_address", 1883, 60)

# Process violations asynchronously
def process_violations():
    while True:
        track_id, punch_time = violation_queue.get()
        alert_msg = f"⚠️ Alert: User {track_id} punched but didn't cross!"
        with db_lock:
            cursor.execute("INSERT INTO Violations VALUES (?, ?, ?)", (track_id, punch_time, "Did not cross"))
            conn.commit()
        # mqtt_client.publish("alerts/violations", alert_msg)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=alert_msg)
        violation_queue.task_done()

threading.Thread(target=process_violations, daemon=True).start()

# Webcam (Windows compatible)
cap = cv2.VideoCapture(0)  # Use webcam (0 for default camera)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    results = model(frame, verbose=False)
    detections = []

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        if int(cls) == 0:  # Assuming class 0 is "person"
            detections.append([[x1, y1, x2, y2], conf])  # Format: [[x1, y1, x2, y2], conf]

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_tlbr()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if PUNCHING_ZONE[0] < center_x < PUNCHING_ZONE[2] and PUNCHING_ZONE[1] < center_y < PUNCHING_ZONE[3]:
            if track_id not in user_tracking:
                user_tracking[track_id] = {"punched": datetime.now(), "crossed": False}

        if CROSSING_ZONE[0] < center_x < CROSSING_ZONE[2] and CROSSING_ZONE[1] < center_y < CROSSING_ZONE[3]:
            if track_id in user_tracking:
                user_tracking[track_id]["crossed"] = True

    for track_id, data in list(user_tracking.items()):
        if data["punched"] and not data["crossed"]:
            elapsed_time = (datetime.now() - data["punched"]).seconds
            if elapsed_time > 15:
                violation_queue.put((track_id, data["punched"]))

    cv2.imshow("Live CCTV Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
