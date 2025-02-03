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
# Environment Variables & Logging Setup
# -------------------------------
# Optionally load environment variables from a .env file.
# Ensure that your .env file is added to .gitignore to avoid exposing secrets.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If python-dotenv is not installed, ensure that environment variables are set

# Configure logging to output to both a file and the console.
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'punching_detection.log'
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logging.info("Logging is set up.")

# -------------------------------
# Configuration Constants
# -------------------------------
CONFIDENCE_THRESHOLD = 0.7  # Adjust between 0.5 - 0.7 as needed
LINE_THRESHOLD = 30          # Distance (pixels) to consider "on" a line
FRAME_SKIP = 2               # Process every Nth frame for performance optimization
VIOLATION_DELAY = 10         # Seconds before considering a user as violation

# Enable OpenCV optimizations
cv2.setUseOptimized(True)

# -------------------------------
# Choose Input Mode: Video File or Webcam
# -------------------------------
mode = input("Enter '1' for webcam or '2' for video file: ")
if mode == '2':
    VIDEO_FILE = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(VIDEO_FILE)
else:
    cap = cv2.VideoCapture(0)  # Default webcam

# -------------------------------
# Frame Reader Thread
# -------------------------------
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
# Main Initialization and Setup
# -------------------------------

# Enable GPU acceleration if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"[INFO] Using device: {device}")

# Load YOLOv8 model
from ultralytics import YOLO
model = YOLO("yolov8m.pt").to(device)
# logging.info("[INFO] YOLOv8 model loaded successfully.")

# Initialize DeepSORT tracker
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=70, n_init=3, nn_budget=100)
# logging.info("[INFO] DeepSORT tracker initialized.")

# Telegram bot setup using environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("Telegram bot credentials are not set in environment variables.")
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
# logging.info("[INFO] Telegram bot initialized.")

# SQLite database connection
conn = sqlite3.connect("violations.db", check_same_thread=False)
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
db_lock = Lock()
logging.info("[INFO] Database connected and table ensured.")

# -------------------------------
# Select Punching & Crossing Lines
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
# Tracking & Violation Storage
# -------------------------------
user_tracking = {}       # Format: { track_id: {"punched": datetime, "crossed": bool} }
violations_recorded = {}  # Format: { track_id: date }
violation_queue = Queue(maxsize=100)

# -------------------------------
# Ensure Violation Frames Directory Exists
# -------------------------------
VIOLATION_DIR = "violation_frames"
os.makedirs(VIOLATION_DIR, exist_ok=True)

# -------------------------------
# Notification Functions
# -------------------------------
# Email Configurations (loaded from environment variables)
EMAIL_CONFIG = {
    "sender_email": os.getenv("EMAIL_SENDER"),
    "receiver_email": os.getenv("EMAIL_RECEIVER"),
    "password": os.getenv("EMAIL_PASSWORD"),
}
if not EMAIL_CONFIG["sender_email"] or not EMAIL_CONFIG["receiver_email"] or not EMAIL_CONFIG["password"]:
    logging.error("Email configuration is incomplete. Please set EMAIL_SENDER, EMAIL_RECEIVER, and EMAIL_PASSWORD.")

# Twilio Configurations (loaded from environment variables)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER")
# if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and TWILIO_TO_NUMBER):
#    logging.error("Twilio configuration is incomplete. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, and TWILIO_TO_NUMBER.")

# WhatsApp Configuration (loaded from environment variables with defaults)
WHATSAPP_URL = os.getenv("WHATSAPP_URL", "http://192.168.68.112:5525/api/v1/sendMessage")
WHATSAPP_ACCOUNT_ID = os.getenv("WHATSAPP_ACCOUNT_ID")
WHATSAPP_TO = os.getenv("WHATSAPP_TO")
WHATSAPP_MSG_TYPE = os.getenv("WHATSAPP_MSG_TYPE", "attendance")
if not WHATSAPP_ACCOUNT_ID or not WHATSAPP_TO:
    logging.error("WhatsApp configuration is incomplete. Please set WHATSAPP_ACCOUNT_ID and WHATSAPP_TO.")

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
            logging.error(f"Failed to send WhatsApp message. Status Code: {response.status_code}, Response: {response.text}")
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
    Returns a dictionary with the responses from each channel.
    """
    # Send SMS (uncomment the line below to enable SMS sending)
    sms_sid = "123456" # send_sms(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER, alert_msg)
    logging.info(f"SMS sent with SID: {sms_sid}")

    # Send WhatsApp message
    whatsapp_response = send_custom_message(alert_msg)
    logging.info(f"WhatsApp response: {whatsapp_response}")

    # Send Telegram message
    try:
        telegram_response = bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                               text=alert_msg + f" (Frame saved at {image_path})")
        logging.info(f"Telegram response: {telegram_response}")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")
        telegram_response = None

    # Send Email
    email_subject = "Violation Alert Notification"
    email_body = alert_msg + f"\nFrame saved at: {image_path}"
    email_response = send_email(EMAIL_CONFIG["sender_email"],
                                EMAIL_CONFIG["receiver_email"],
                                EMAIL_CONFIG["password"],
                                email_subject,
                                email_body)
    logging.info(f"Email response: {email_response}")

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
        logging.info(f"[VIOLATION] Processing violation: {alert_msg}")

        # Save the violation frame with a unique filename
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
# Main Processing Loop
# -------------------------------
frame_count = 0
while True:
    frame = frame_queue.get()
    if frame is None:  # Sentinel received, end of stream
        logging.info("[INFO] End of video stream. Exiting...")
        break

    # Only process every FRAME_SKIP-th frame for performance
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # Draw the punching (red) and crossing (green) lines on the frame
    cv2.line(frame, punching_line[0], punching_line[1], (0, 0, 255), 2)
    cv2.line(frame, crossing_line[0], crossing_line[1], (0, 255, 0), 2)

    # Get current time once for use later (avoids multiple datetime.now() calls)
    now = datetime.now()
    current_date = now.date()

    # Run YOLOv8 detection on the current frame
    results = model(frame, verbose=False)
    detections = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            # Only consider persons (class 0) with sufficient confidence
            if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box
                # Draw detection rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 20, 147), 2)
                detections.append([[x1, y1, x2, y2], conf])
                logging.info(f"[DETECTION] Person detected at: {(x1, y1, x2, y2)} with confidence {conf:.2f}")

    # Update tracks using DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)
    logging.info(f"[INFO] Number of active tracks: {len(tracks)}")

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = track.to_tlbr()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Compute the distances once for efficiency
        distance_to_punch = point_line_distance((center_x, center_y), punching_line)
        distance_to_cross = point_line_distance((center_x, center_y), crossing_line)

        # Determine the color of the ID text based on proximity (crossing has higher priority)
        id_color = (255, 0, 255)  # Magenta
        if distance_to_punch < LINE_THRESHOLD:
            id_color = (255, 165, 0)  # Orange
            logging.info("Center is near the punching line")

            # Register punch event if within threshold and not already recorded
            if track_id not in user_tracking:
                user_tracking[track_id] = {"punched": now, "crossed": False}
                logging.info(f"[PUNCH] User {track_id} punched at {now} (distance: {distance_to_punch:.2f})")

        if distance_to_cross < LINE_THRESHOLD:
            id_color = (0, 255, 127)  # Spring Green
            logging.info("Center is near the crossing line")

            # Register crossing event if within threshold and not yet marked
            if track_id in user_tracking and not user_tracking[track_id]["crossed"]:
                user_tracking[track_id]["crossed"] = True
                logging.info(f"[CROSS] User {track_id} successfully crossed (distance: {distance_to_cross:.2f})")

        # Draw the tracked ID text on the frame
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)

    # Check for violations: users who were punched but have not crossed within a delay period
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

    logging.info(f"[QUEUE] Current Violation Queue Size: {violation_queue.qsize()}")
    cv2.imshow("Live CCTV Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("[INFO] Exiting...")
        break
        
# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
conn.close()
logging.info("[INFO] Cleanup completed.")