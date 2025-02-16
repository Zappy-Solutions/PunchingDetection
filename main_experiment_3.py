import os
import cv2
import torch
import numpy as np
import sqlite3
import threading
import logging
import requests
import smtplib
import shutil
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from queue import Queue, Empty
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
CONFIDENCE_THRESHOLD = 0.75  # Adjust as needed
LINE_THRESHOLD = 5  # Distance in pixels to consider a point "on" the line
FRAME_SKIP = 5  # (For file processing: process every Nth frame)
VIOLATION_DELAY = 60  # Seconds before considering a user as in violation

# Video stream parameters (for recording)
window_width = 1280
window_height = 720
fps = 25  # Frames per second
SEGMENT_DURATION = 60  # Duration (in seconds) for each video segment

# Choose input mode: RTSP, webcam, or video file
# Hikvision
# rtsp_url = "rtsp://admin:525ForgetMe!@192.168.68.118:554/Streaming/Channels/101/?tcp"

#4MP Impact
rtsp_url = "rtsp://admin:admin123@192.168.68.124:554/rtsp/streaming?channel=01&subtype=0"

mode = input("Enter '1' for RTSP, '2' for webcam, or '3' for video file: ").strip()
if mode == '1':
    logging.info(f"Selected RTSP mode with URL: {rtsp_url}")
    STREAM_SOURCE = rtsp_url
elif mode == '2':
    logging.info("Selected webcam mode")
    STREAM_SOURCE = 0  # default webcam
else:
    VIDEO_FILE = input("Enter the path to the video file: ").strip()
    logging.info(f"Selected video file mode with file: {VIDEO_FILE}")
    STREAM_SOURCE = VIDEO_FILE

# -------------------------------
# Directories for File Segmentation
# -------------------------------
INCOMING_DIR = "incoming_files"
PROCESSED_DIR = "processed_files"
os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------------
# Global Shared Structures
# -------------------------------
# For violation processing:
violation_queue = Queue()
user_tracking = {}  # Format: { track_id: {"punched": datetime, "crossed": bool} }
violations_recorded = {}  # Format: { track_id: date }
db_lock = Lock()  # For database access


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
    return send_message_http(WHATSAPP_ACCOUNT_ID, WHATSAPP_TO, message, WHATSAPP_MSG_TYPE)


def send_email(sender_email, receiver_email, password, subject, body):
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
        res = send_email(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["receiver_email"],
                         EMAIL_CONFIG["password"], subject, body)
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
# Line Selection with Full Width
# -------------------------------
def select_line_full(window_name, frame):
    """
    Allows the user to select a full-width horizontal line by clicking two points.
    The y-coordinate of the first click is used for the full-width line.
    """
    points = []
    clone = frame.copy()
    height, width, _ = frame.shape

    def mouse_callback(event, x, y, flags, param):
        nonlocal clone, points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, clone)
        if len(points) == 2:
            y_selected = points[0][1]
            points[:] = [(0, y_selected), (width, y_selected)]
            clone = frame.copy()
            cv2.line(clone, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow(window_name, clone)

    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_callback)
    print(f"Click two points to set the {window_name}. Press 'c' to confirm.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(points) == 2:
            break
    cv2.setMouseCallback(window_name, lambda *args: None)
    cv2.destroyWindow(window_name)
    logging.info(f"[INFO] Selected {window_name}: {points[0]} to {points[1]}")
    return points[0], points[1]


# -------------------------------
# Recording & Processing Setup
# -------------------------------

# (1) Get an initial frame for line selection.
# Open a temporary capture to get a frame.
temp_cap = cv2.VideoCapture(STREAM_SOURCE)
ret, init_frame = temp_cap.read()
temp_cap.release()
if not ret:
    logging.error("[ERROR] Could not retrieve initial frame from the stream.")
    exit()

punching_line = select_line_full("Select Punching Line", init_frame.copy())
crossing_line = select_line_full("Select Crossing Line", init_frame.copy())


# -------------------------------
# Recording Thread: Segment the Live Stream into Files
# -------------------------------
def record_video():
    # Open a new capture for recording
    cap_rec = cv2.VideoCapture(STREAM_SOURCE)
    cap_rec.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap_rec.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
    cap_rec.set(cv2.CAP_PROP_FPS, fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    while cap_rec.isOpened():
        start_time = datetime.now()
        filename = start_time.strftime("%Y%m%d_%H%M%S") + ".avi"
        filepath = os.path.join(INCOMING_DIR, filename)
        out = cv2.VideoWriter(filepath, fourcc, fps, (window_width, window_height))
        logging.info(f"[RECORD] Recording new segment: {filename}")
        while (datetime.now() - start_time).total_seconds() < SEGMENT_DURATION:
            ret, frame = cap_rec.read()
            if not ret:
                break
            out.write(frame)
            # (Optional) Show live recording preview.
            cv2.imshow("Live Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap_rec.release()
                out.release()
                cv2.destroyAllWindows()
                return
        out.release()
        logging.info(f"[RECORD] Finished recording segment: {filename}")
    cap_rec.release()
    cv2.destroyAllWindows()


# -------------------------------
# Processing Thread: Process Video Files from Incoming Folder
# -------------------------------
def process_video_files():
    """
    Monitors the INCOMING_DIR for new video segments.
    For each file, processes it frame-by-frame using detection and tracking
    (while maintaining the persistent tracker) and then moves the file to PROCESSED_DIR.
    """
    global tracker  # use the persistent tracker instance
    while True:
        files = sorted(os.listdir(INCOMING_DIR))
        if not files:
            time.sleep(1)
            continue

        for filename in files:
            filepath = os.path.join(INCOMING_DIR, filename)
            logging.info(f"[PROCESS] Processing file: {filename}")
            cap_file = cv2.VideoCapture(filepath)
            frame_count = 0
            while True:
                ret, frame = cap_file.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                # Draw the static lines
                cv2.line(frame, punching_line[0], punching_line[1], (0, 0, 255), 2)
                cv2.line(frame, crossing_line[0], crossing_line[1], (0, 255, 0), 2)

                now = datetime.now()
                # --- Detection ---
                results = model(frame, verbose=False)
                detections = []
                for result in results:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 20, 147), 2)
                            detections.append([[x1, y1, x2, y2], conf])
                            logging.info(
                                f"[DETECTION] Person detected at: {(x1, y1, x2, y2)} with confidence {conf:.2f}")

                # --- Tracking & Drawing ---
                tracks = tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    x1, y1, x2, y2 = track.to_tlbr()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    bbox_points = [
                        (x1, y1),
                        (x2, y1),
                        (x1, y2),
                        (x2, y2),
                        (center_x, center_y)
                    ]
                    punched = any(point_line_distance(pt, punching_line) < LINE_THRESHOLD for pt in bbox_points)
                    crossed = any(point_line_distance(pt, crossing_line) < LINE_THRESHOLD for pt in bbox_points)
                    logging.info(f"[TRACK] punched: {punched} \t crossed: {crossed}")

                    id_color = (255, 0, 255)  # default magenta
                    if crossed:
                        id_color = (0, 255, 127)  # spring green
                        if track_id in user_tracking and not user_tracking[track_id]["crossed"]:
                            user_tracking[track_id]["crossed"] = True
                            if "punched" not in user_tracking[track_id]:
                                user_tracking[track_id] = {"punched": now}
                            logging.info(f"[CROSS] User {track_id} successfully crossed")
                    if punched:
                        id_color = (255, 165, 0)  # orange
                        if track_id not in user_tracking:
                            user_tracking[track_id] = {"punched": now, "crossed": False}
                            logging.info(f"[PUNCH] User {track_id} punched at {now}")

                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)

                # --- Violation Check ---
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

                # Optionally display the processed frame for debugging
                cv2.namedWindow("Processed Segment", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Processed Segment", window_width, window_height)
                cv2.imshow("Processed Segment", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap_file.release()
            # After processing, move the file to the processed folder.
            shutil.move(filepath, os.path.join(PROCESSED_DIR, filename))
            logging.info(f"[PROCESS] Moved {filename} to {PROCESSED_DIR}")
        time.sleep(1)
    cv2.destroyAllWindows()


# -------------------------------
# Start the Recording and Processing Threads
# -------------------------------
record_thread = threading.Thread(target=record_video, daemon=True)
process_thread = threading.Thread(target=process_video_files, daemon=True)

record_thread.start()
process_thread.start()

# Keep the main thread alive.
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logging.info("[INFO] Exiting...")