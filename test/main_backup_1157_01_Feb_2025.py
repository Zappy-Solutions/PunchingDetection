import os
import cv2
import torch
import numpy as np
import sqlite3
import threading
import logging
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from queue import Queue
import telegram
from threading import Lock
from twilio.rest import Client

# -------------------------------
# Configuration
# -------------------------------
CONFIDENCE_THRESHOLD = 0.65  # Adjust between 0.5 - 0.7 as needed
LINE_THRESHOLD = 30  # Distance (pixels) to consider "on" a line
FRAME_SKIP = 2  # Process every Nth frame for performance optimization
VIOLATION_DELAY = 10  # Seconds before considering a user as violation

# Enable OpenCV optimizations
cv2.setUseOptimized(True)

# -------------------------------
# Choose Input Mode: Video File or Webcam
# -------------------------------
mode = input("Enter '1' for webcam or '2' for video file: ")
if mode == '2':
    VIDEO_FILE = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(VIDEO_FILE)
    # Example: cap = cv2.VideoCapture("Punching.mp4")
else:
    cap = cv2.VideoCapture(0)  # Default webcam

# -------------------------------
# Frame Reader Thread
# -------------------------------
# Create a thread-safe queue to hold frames.
frame_queue = Queue(maxsize=5)


def read_frames(cap, queue):
    """
    Continuously read frames from the capture and put them into the queue.
    When the stream ends, put a sentinel (None) into the queue.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        queue.put(frame)
    queue.put(None)


# Start the frame reader thread (daemon thread so it exits when the main program exits)
threading.Thread(target=read_frames, args=(cap, frame_queue), daemon=True).start()


# -------------------------------
# Helper Functions
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
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
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

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(points) >= 2:
            break
    cv2.setMouseCallback(window_name, lambda *args: None)  # Disable callback
    return points[0], points[1]


# -------------------------------
# Main Initialization and Setup
# -------------------------------

# Enable GPU acceleration if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# Load YOLOv8 model
# print("[INFO] Loading YOLOv8 model...")
from ultralytics import YOLO

model = YOLO("yolov8m.pt").to(device)
# print("[INFO] YOLOv8 model loaded successfully.")

# Initialize DeepSORT tracker (Real-time version)
# print("[INFO] Initializing DeepSORT tracker...")
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=50, n_init=3, nn_budget=100)

# Telegram bot setup
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
# print("[INFO] Telegram bot initialized.")

# SQLite database connection
conn = sqlite3.connect("violations.db", check_same_thread=False)
cursor = conn.cursor()

# Drop the table if it already exists to update schema (remove this line if you want to preserve existing data)
# cursor.execute("DROP TABLE IF EXISTS Violations")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS Violations (
        UserID TEXT,
        Time TEXT,
        Issue TEXT,
        ImagePath TEXT
    )
""")
# Enable Write-Ahead Logging (WAL) mode for better performance
conn.execute("PRAGMA journal_mode=WAL;")
db_lock = Lock()
print("[INFO] Database connected and table ensured.")

# -------------------------------
# Select Punching & Crossing Lines
# -------------------------------
# Use an initial frame (read directly from cap) for line selection
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
user_tracking = {}  # Format: { track_id: {"punched": datetime, "crossed": bool} }
violations_recorded = {}  # Format: { track_id: date }
violation_queue = Queue(maxsize=100)

# MQTT setup for real-time alerting (currently commented out)
# mqtt_client = mqtt.Client()
# mqtt_client.connect("mqtt_broker_address", 1883, 60)

# -------------------------------
# Ensure Violation Frames Directory Exists
# -------------------------------
VIOLATION_DIR = "violation_frames"
os.makedirs(VIOLATION_DIR, exist_ok=True)

# -------------------------------
# Notification Functions
# -------------------------------

# Email Configurations
EMAIL_CONFIG = {
    "sender_email": "canucatch@gmail.com",
    "receiver_email": "yashkysolutions@gmail.com",
    "password": "vzgq ypvt ozpm vsqh",
}

# Twilio Configurations for SMS
TWILIO_ACCOUNT_SID = 'AC3244cdef4a8f291825c0e030a6f4940c'
TWILIO_AUTH_TOKEN = 'f7b39cf87d81c00db2143131f0d5cdea'
TWILIO_FROM_NUMBER = '+15184056680'
TWILIO_TO_NUMBER = '+919148978095'

# WhatsApp Configuration parameters (via HTTP POST)
WHATSAPP_URL = "http://192.168.68.112:5525/api/v1/sendMessage"
WHATSAPP_ACCOUNT_ID = "6752cd26017946b0e"
WHATSAPP_TO = "919148978095@c.us"
WHATSAPP_MSG_TYPE = "attendance"


def send_sms(account_sid, auth_token, from_number, to_number, message_body):
    """
    Sends an SMS using the Twilio API.
    """
    print("Sending SMS notification...")
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=to_number
        )
        return message.sid
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return None


def send_message_http(account_id, to, message, msg_type):
    """
    Sends a message via HTTP POST request (for WhatsApp notifications).
    """
    payload = {
        "accountId": account_id,
        "to": to,
        "message": message,
        "type": msg_type
    }
    try:
        response = requests.post(WHATSAPP_URL, json=payload)
        if response.status_code == 200:
            logging.info("WhatsApp message sent successfully")
            return response.json()
        else:
            logging.error(
                f"Failed to send WhatsApp message. Status Code: {response.status_code}, Response: {response.text}")
            return {"error": response.text}
    except Exception as e:
        logging.error(f"Exception occurred while sending WhatsApp message: {str(e)}")
        return {"error": str(e)}


def send_custom_message(message):
    """
    Sends a custom message via WhatsApp.
    """
    return send_message_http(WHATSAPP_ACCOUNT_ID, WHATSAPP_TO, message, WHATSAPP_MSG_TYPE)


def send_email(sender_email, receiver_email, password, subject, body):
    """
    Sends an email using SMTP.
    """
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return "Email sent successfully"
    except Exception as e:
        return f"Failed to send email: {e}"


def send_notifications(alert_msg, image_path):
    """
    Sends notifications via SMS, WhatsApp, Telegram, and Email.
    Returns a dictionary with the responses from each channel.
    """
    # Send SMS
    sms_sid = "123456" # send_sms(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER, alert_msg)
    print(f"SMS sent with SID: {sms_sid}")

    # Send WhatsApp message
    whatsapp_response = send_custom_message(alert_msg)
    print(f"WhatsApp response: {whatsapp_response}")

    # Send Telegram message
    try:
        telegram_response = bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                             text=alert_msg + f" (Frame saved at {image_path})")
        print(f"Telegram response: {telegram_response}")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        telegram_response = None

    # Send Email
    email_subject = "Violation Alert Notification"
    email_body = alert_msg + f"\nFrame saved at: {image_path}"
    email_response = send_email(EMAIL_CONFIG["sender_email"],
                                EMAIL_CONFIG["receiver_email"],
                                EMAIL_CONFIG["password"],
                                email_subject,
                                email_body)
    print(f"Email response: {email_response}")

    return {
        "sms": sms_sid,
        "whatsapp": whatsapp_response,
        "telegram": telegram_response,
        "email": email_response
    }


# -------------------------------
# Violation Processing Thread
# -------------------------------
def process_violations():
    while True:
        # Expecting a tuple: (track_id, punch_time, violation_frame)
        track_id, punch_time, violation_frame = violation_queue.get()
        alert_msg = f"⚠️ Alert: User {track_id} punched but didn't cross!"
        print(f"[VIOLATION] Processing violation: {alert_msg}")

        # Save the violation frame with a unique filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(VIOLATION_DIR, f"violation_{track_id}_{timestamp_str}.jpg")
        cv2.imwrite(image_path, violation_frame)

        with db_lock:
            # Format time to show only up to seconds
            time_str = punch_time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO Violations VALUES (?, ?, ?, ?)",
                           (track_id, time_str, "Did not cross", image_path))
            conn.commit()

        # Send notifications via SMS, WhatsApp, Telegram, and Email
        responses = send_notifications(alert_msg, image_path)
        print(f"[ALERT] Notifications sent for User {track_id}: {responses}")
        violation_queue.task_done()


threading.Thread(target=process_violations, daemon=True).start()
print("[INFO] Violation processing thread started.")

# -------------------------------
# Main Processing Loop
# -------------------------------
frame_count = 0
while True:
    frame = frame_queue.get()
    if frame is None:  # Sentinel received, end of stream
        print("[INFO] End of video stream. Exiting...")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Skip frames for performance

    # Draw the selected punching and crossing lines on the frame
    cv2.line(frame, punching_line[0], punching_line[1], (0, 0, 255), 2)  # Red line for punching
    cv2.line(frame, crossing_line[0], crossing_line[1], (0, 255, 0), 2)  # Green line for crossing

    # Run YOLOv8 detection on the current frame
    results = model(frame, verbose=False)
    detections = []
    print("[INFO] Processing detections...")

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = box  # Bounding box coordinates
            if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD:
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

        # Determine ID color based on proximity to lines
        if point_line_distance((center_x, center_y), punching_line) < LINE_THRESHOLD:
            print("Center is near the punching line")
            id_color = (255, 165, 0)  # Orange
        elif point_line_distance((center_x, center_y), crossing_line) < LINE_THRESHOLD:
            print("Center is near the crossing line")
            id_color = (0, 255, 127)  # Spring Green
        else:
            id_color = (255, 0, 255)  # Default: Magenta

        # Draw the tracked ID text on the frame
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)

        # Check if the center is near the punching line.
        distance_to_punch = point_line_distance((center_x, center_y), punching_line)
        if distance_to_punch < LINE_THRESHOLD:
            if track_id not in user_tracking:
                user_tracking[track_id] = {"punched": datetime.now(), "crossed": False}
                print(
                    f"[PUNCH] User {track_id} punched at {user_tracking[track_id]['punched']} (distance: {distance_to_punch:.2f})")

        # Check if the center is near the crossing line.
        distance_to_cross = point_line_distance((center_x, center_y), crossing_line)
        if distance_to_cross < LINE_THRESHOLD:
            if track_id in user_tracking and not user_tracking[track_id]["crossed"]:
                user_tracking[track_id]["crossed"] = True
                print(f"[CROSS] User {track_id} successfully crossed (distance: {distance_to_cross:.2f})")

    # Check for users who have been punched but not crossed after a delay
    for track_id, data in list(user_tracking.items()):
        if data["punched"] and not data["crossed"]:
            elapsed_time = (datetime.now() - data["punched"]).seconds
            if elapsed_time > VIOLATION_DELAY:
                current_date = datetime.now().date()
                if track_id in violations_recorded and violations_recorded[track_id] == current_date:
                    print(f"[INFO] User {track_id} already recorded for today, skipping.")
                    continue
                print(f"[VIOLATION] Adding User {track_id} to violation queue.")
                # Save the current frame as the violation frame
                violation_queue.put((track_id, data["punched"], frame.copy()))
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