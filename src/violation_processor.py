# violation_processor.py
import os
import cv2
import logging
from datetime import datetime
from database import db_lock
from notifications import send_notifications

# Ensure the directory for violation frames exists
VIOLATION_DIR = "violation_frames"
os.makedirs(VIOLATION_DIR, exist_ok=True)

def process_violations(violation_queue, cursor, conn):
    """
    Processes violations by saving the violation frame, updating the database,
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