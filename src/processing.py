# processing.py
import cv2
import logging
from datetime import datetime
from utils import point_line_distance
from tracker import model, tracker
from config import CONFIDENCE_THRESHOLD, LINE_THRESHOLD, VIOLATION_DELAY


def process_detections(frame):
    """
    Runs YOLOv8 detection on the frame, draws detection boxes,
    and returns a list of detections.
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


def update_tracks_and_draw(frame, detections, now, punching_line, crossing_line, user_tracking):
    """
    Updates tracks using DeepSORT, draws track IDs on the frame,
    and checks for punching and crossing events.
    """
    tracks = tracker.update_tracks(detections, frame=frame)
    logging.info(f"[INFO] Number of active tracks: {len(tracks)}")

    for track in tracks:
        track_id = track.track_id

        # Ensure track_id is initialized in user_tracking
        if track_id not in user_tracking:
            user_tracking[track_id] = {"punched": None, "crossed": None}  # Store timestamps

        # **Skip unconfirmed tracks**
        if not track.is_confirmed():
            logging.warning(f"track_id: {track_id} is not confirmed, skipping.")
            continue

        # **Skip tracks that have already crossed**
        if user_tracking[track_id]["crossed"]:
            # logging.warning(f"track_id: {track_id} is already crossed, skipping processing.")
            continue

        x1, y1, x2, y2 = track.to_tlbr()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Define multiple points on the bounding box
        bbox_points = [
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2),
            (center_x, center_y)
        ]

        punched = any(point_line_distance(pt, punching_line) < LINE_THRESHOLD for pt in bbox_points)
        crossed = any(point_line_distance(pt, crossing_line) < LINE_THRESHOLD for pt in bbox_points)

        logging.info(f"track_id: {track_id} \t punched: {punched} \t crossed: {crossed}")

        id_color = (255, 0, 255)  # Default magenta

        if crossed:
            id_color = (0, 255, 127)  # Spring green
            logging.info(f"track_id: {track_id} \t Center is near the crossing line")

            # Only update `crossed` if `track_id` exists in user_tracking
            if user_tracking[track_id]["crossed"] is None:
                user_tracking[track_id]["crossed"] = now  # Store the timestamp

                # If user crossed first, still ensure punched has a valid timestamp
                if user_tracking[track_id]["punched"] is None:
                    user_tracking[track_id]["punched"] = now  # Set punched time to crossing time if missing

                logging.info(f"[CROSS] User {track_id} successfully crossed at {now}")

        elif punched:
            id_color = (0, 165, 255)  # Orange
            logging.info(f"track_id: {track_id} \t Center is near the punching line")

            logging.info(
                f"track_id: {track_id} \t punched: {user_tracking[track_id].get('punched', 'None')} \t crossed: {user_tracking[track_id].get('crossed', 'None')}")

            if user_tracking[track_id]["punched"] is None:
                user_tracking[track_id]["punched"] = now  # Store the timestamp
                logging.info(f"[PUNCH] User {track_id} punched at {now}")

        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)


def check_violations(frame, now, user_tracking, violations_recorded, violation_queue):
    """
    Checks for users who have been punched but not crossed within the delay period.
    If a violation is detected, it is added to the violation queue.
    """
    current_date = now.date()
    for track_id, data in list(user_tracking.items()):
        if data["punched"] is not None and data["crossed"] is None:
            elapsed_time = (now - data["punched"]).seconds
            if elapsed_time > VIOLATION_DELAY:
                if violations_recorded.get(track_id) == current_date:
                    logging.info(f"[INFO] User {track_id} already recorded for today, skipping.")
                    continue
                logging.info(f"[VIOLATION] Adding User {track_id} to violation queue.")
                violation_queue.put((track_id, data["punched"], frame.copy()))
                violations_recorded[track_id] = current_date