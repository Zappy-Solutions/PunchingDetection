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
from concurrent.futures import ThreadPoolExecutor

# -------------------------------
# Environment Variables & Logging Setup
# -------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # Make sure environment variables are set externally


def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = 'punching_detection.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    logging.info("Logging is set up.")


setup_logging()

# -------------------------------
# Configuration Constants
# -------------------------------
CONFIDENCE_THRESHOLD = 0.8  # Adjust between 0.5 - 0.7 as needed
LINE_THRESHOLD = 5  # Distance (pixels) to consider "on" a line
FRAME_SKIP = 5  # Process every Nth frame for performance
VIOLATION_DELAY = 60  # Seconds before considering a user as violation

cv2.setUseOptimized(True)

# -------------------------------
# Input Mode: Video File or Webcam
# -------------------------------
# rtsp_url = "rtsp://admin:admin@192.168.68.113:554/streaming?channel=01&subtype=A"
# rtsp_url = "rtsp://192.168.68.113:554/streaming?channel=01&subtype=0"
# rtsp_url = "rtsp://admin:admin123@192.168.68.114:554/streaming?channel=01&subtype=0"

rtsp_url = "rtsp://192.168.68.113:554/rtsp/streaming?channel=01&subtype=0"

mode = input("Enter '1' for RTSP, '2' for webcam, or '3' for video file: ")
if mode == '1':
    # rtsp_url = input("Enter the RTSP URL: ")
    logging.info(f"Selected RTSP mode with URL: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    # Set resolution to 480p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    desired_fps = 10  # Change this as needed
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"FPS: {fps}")

elif mode == '2':
    logging.info("Selected webcam mode")
    cap = cv2.VideoCapture(0)  # Default webcam
else:
    logging.info("Selected video file mode")
    VIDEO_FILE = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(VIDEO_FILE)

# -------------------------------
# Global Queues and Shared Structures
# -------------------------------
frame_queue = Queue(maxsize=5)
violation_queue = Queue(maxsize=100)
user_tracking = {}  # { track_id: {"punched": datetime, "crossed": bool} }
violations_recorded = {}  # { track_id: date }

# Lock for database and other shared resources
db_lock = Lock()


# -------------------------------
# Database Setup
# -------------------------------
def setup_database(db_path="violations.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Violations (
            UserID TEXT,
            Time TEXT,
            Issue TEXT,
            ImagePath TEXT
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL;")
    logging.info("[INFO] Database connected and table ensured.")
    return conn, cursor


conn, cursor = setup_database()


# -------------------------------
# Frame Reading Thread
# -------------------------------
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
    logging.info("[INFO] Frame reader thread ended.")


threading.Thread(target=read_frames, args=(cap, frame_queue), daemon=True).start()


# -------------------------------
# Helper Functions
# -------------------------------
def point_line_distance(point, line):
    """
    Calculate the perpendicular distance from a point to a line.
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
# Model & Tracker Initialization
# -------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"[INFO] Using device: {device}")

from ultralytics import YOLO

model = YOLO("yolov8m.pt").to(device)
logging.info("[INFO] YOLOv8 model loaded successfully.")

from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=70, n_init=3, nn_budget=100)
logging.info("[INFO] DeepSORT tracker initialized.")

# -------------------------------
# Telegram Bot Setup
# -------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("Telegram bot credentials are not set in environment variables.")
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
logging.info("[INFO] Telegram bot initialized.")

# -------------------------------
# Notification Functions
# -------------------------------
EMAIL_CONFIG = {
    "sender_email": os.getenv("EMAIL_SENDER"),
    "receiver_email": os.getenv("EMAIL_RECEIVER"),
    "password": os.getenv("EMAIL_PASSWORD"),
}
if not EMAIL_CONFIG["sender_email"] or not EMAIL_CONFIG["receiver_email"] or not EMAIL_CONFIG["password"]:
    logging.error("Email configuration is incomplete.")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER")

WHATSAPP_URL = os.getenv("WHATSAPP_URL", "http://192.168.68.112:5525/api/v1/sendMessage")
WHATSAPP_ACCOUNT_ID = os.getenv("WHATSAPP_ACCOUNT_ID")
WHATSAPP_TO = os.getenv("WHATSAPP_TO")
WHATSAPP_MSG_TYPE = os.getenv("WHATSAPP_MSG_TYPE", "attendance")
if not WHATSAPP_ACCOUNT_ID or not WHATSAPP_TO:
    logging.error("WhatsApp configuration is incomplete.")


def send_sms(account_sid, auth_token, from_number, to_number, message_body):
    """Sends an SMS using the Twilio API."""
    logging.info("Sending SMS notification...")
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=to_number
        )
        return message.sid
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")
        return None


def send_message_http(account_id, to, message, msg_type):
    """Sends a message via HTTP POST request (for WhatsApp notifications)."""
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
    """Sends a custom message via WhatsApp."""
    return send_message_http(WHATSAPP_ACCOUNT_ID, WHATSAPP_TO, message, WHATSAPP_MSG_TYPE)


def send_email(sender_email, receiver_email, password, subject, body):
    """Sends an email using SMTP."""
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
    Uses a thread pool to send notifications concurrently.
    """
    responses = {}

    def sms_task():
        sid = send_sms(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER, alert_msg)
        responses["sms"] = sid
        logging.info(f"SMS sent with SID: {sid}")

    def whatsapp_task():
        res = send_custom_message(alert_msg)
        responses["whatsapp"] = res
        logging.info(f"WhatsApp response: {res}")

    def telegram_task():
        try:
            res = bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                   text=alert_msg + f" (Frame saved at {image_path})")
            responses["telegram"] = res
            logging.info(f"Telegram response: {res}")
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")
            responses["telegram"] = None

    def email_task():
        subject = "Violation Alert Notification"
        body = alert_msg + f"\nFrame saved at: {image_path}"
        res = send_email(EMAIL_CONFIG["sender_email"],
                         EMAIL_CONFIG["receiver_email"],
                         EMAIL_CONFIG["password"],
                         subject, body)
        responses["email"] = res
        logging.info(f"Email response: {res}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(sms_task)
        executor.submit(whatsapp_task)
        executor.submit(telegram_task)
        executor.submit(email_task)

    return responses


# -------------------------------
# Violation Processing Thread
# -------------------------------
VIOLATION_DIR = "violation_frames"
os.makedirs(VIOLATION_DIR, exist_ok=True)


def process_violations():
    """
    Processes violations by saving the violation frame, writing to the database,
    and sending notifications.
    """
    while True:
        # Expecting a tuple: (track_id, punch_time, violation_frame)
        track_id, punch_time, violation_frame = violation_queue.get()
        alert_msg = f"Alert: User {track_id} punched but didn't cross!"
        logging.info(f"[VIOLATION] Processing violation: {alert_msg}")

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(VIOLATION_DIR, f"violation_{track_id}_{timestamp_str}.jpg")
        cv2.imwrite(image_path, violation_frame)

        with db_lock:
            time_str = punch_time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO Violations VALUES (?, ?, ?, ?)",
                           (track_id, time_str, "Did not cross", image_path))
            conn.commit()

        responses = send_notifications(alert_msg, image_path)
        logging.info(f"[ALERT] Notifications sent for User {track_id}: {responses}")
        violation_queue.task_done()


threading.Thread(target=process_violations, daemon=True).start()
logging.info("[INFO] Violation processing thread started.")

# -------------------------------
# Line Selection (User Input)
# -------------------------------
ret, init_frame = cap.read()
if not ret:
    logging.error("[ERROR] Could not read first frame for line selection.")
    exit()
punching_line = select_line("Select Punching Line", init_frame.copy())
logging.info(f"[INFO] Punching line selected: {punching_line}")
crossing_line = select_line("Select Crossing Line", init_frame.copy())
logging.info(f"[INFO] Crossing line selected: {crossing_line}")

cv2.destroyWindow("Select Punching Line")
cv2.destroyWindow("Select Crossing Line")


# -------------------------------
# Frame Processing Functions
# -------------------------------
def process_detections(frame):
    """
    Run YOLOv8 detection on the frame and return a list of detections.
    Also draws the detection rectangles.
    """
    results = model(frame, verbose=False)
    detections = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 20, 147), 2)
                detections.append([[x1, y1, x2, y2], conf])
                logging.info(f"[DETECTION] Person detected at: {(x1, y1, x2, y2)} with confidence {conf:.2f}")
    return detections


def update_tracks_and_draw(frame, detections, now):
    """
    Update tracks using DeepSORT and update the frame with track IDs and events.
    Also check for punching and crossing events.
    """
    tracks = tracker.update_tracks(detections, frame=frame)
    logging.info(f"[INFO] Number of active tracks: {len(tracks)}")

    for track in tracks:
        if not track.is_confirmed():
            continue

        # track_id = track.track_id
        # x1, y1, x2, y2 = track.to_tlbr()
        # center_x = (x1 + x2) / 2
        # center_y = (y1 + y2) / 2
        #
        # # Compute distances to both lines once
        # distance_to_punch = point_line_distance((center_x, center_y), punching_line)
        # distance_to_cross = point_line_distance((center_x, center_y), crossing_line)

        track_id = track.track_id
        x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Define all bounding box points to check against the punching line
        bbox_points = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x1, y2),  # Bottom-left
            (x2, y2),  # Bottom-right
            (center_x, center_y)  # Center point (Optional, for reference)
        ]

        # Check if any of the bounding box points are near the punching line
        punched = any(point_line_distance(pt, punching_line) < LINE_THRESHOLD for pt in bbox_points)
        crossed = any(point_line_distance(pt, crossing_line) < LINE_THRESHOLD for pt in bbox_points)

        logging.info(f"punched:- {punched} \t crossed:- {crossed}")

        id_color = (255, 0, 255)  # Magenta

        # Determine text color based on proximity: crossing takes priority
        if crossed: # distance_to_cross < LINE_THRESHOLD:
            id_color = (0, 255, 127)  # Spring Green
            logging.info("Center is near the crossing line")
            if track_id in user_tracking and not user_tracking[track_id]["crossed"]:
                user_tracking[track_id]["crossed"] = True
                if "punched" not in user_tracking[track_id]:
                    user_tracking[track_id] = {"punched": now}
                logging.info(f"[CROSS] User {track_id} successfully crossed (distance: {crossed:.2f})")

        if punched: # distance_to_punch < LINE_THRESHOLD:
            id_color = (255, 165, 0)  # Orange
            logging.info("Center is near the punching line")
            if track_id not in user_tracking:
                user_tracking[track_id] = {"punched": now, "crossed": False}
                logging.info(f"[PUNCH] User {track_id} punched at {now} (distance: {punched:.2f})")

        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)


def check_violations(frame, now):
    """
    Checks for users who have been punched but not crossed within the delay period.
    If a violation is detected, adds it to the violation queue.
    """
    current_date = now.date()
    for track_id, data in list(user_tracking.items()):
        if not data["crossed"]:
            elapsed_time = (now - data["punched"]).seconds
            if elapsed_time > VIOLATION_DELAY:
                if violations_recorded.get(track_id) == current_date:
                    logging.info(f"[INFO] User {track_id} already recorded for today, skipping.")
                    continue
                logging.info(f"[VIOLATION] Adding User {track_id} to violation queue.")
                violation_queue.put((track_id, data["punched"], frame.copy()))
                violations_recorded[track_id] = current_date


# -------------------------------
# Main Processing Loop
# -------------------------------
def main_loop():
    frame_count = 0
    while True:
        frame = frame_queue.get()
        if frame is None:  # End of stream sentinel
            logging.info("[INFO] End of video stream. Exiting main loop...")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue  # Skip frames to improve performance

        # Draw the static lines
        cv2.line(frame, punching_line[0], punching_line[1], (0, 0, 255), 2)  # Red line for punching
        cv2.line(frame, crossing_line[0], crossing_line[1], (0, 255, 0), 2)  # Green line for crossing

        # Get a single timestamp for this frame processing
        now = datetime.now()

        # Process detections and update tracker/draw info
        # detections = process_detections(frame)
        # update_tracks_and_draw(frame, detections, now)
        # check_violations(frame, now)

        logging.info(f"[QUEUE] Current Violation Queue Size: {violation_queue.qsize()}")
        cv2.namedWindow("Live CCTV Monitoring", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live CCTV Monitoring", 640, 480)
        cv2.imshow("Live CCTV Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("[INFO] 'q' pressed. Exiting main loop...")
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    logging.info("[INFO] Cleanup completed.")


# -------------------------------
# Run the Main Loop
# -------------------------------
if __name__ == '__main__':
    main_loop()