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

    IMPORTANT:
    DeepSORT expects each detection in the format:
        [ [x1, y1, x2, y2], confidence, class ]
    (all as floats, with class typically as an int)
    """
    results = model(frame, verbose=False)
    detections = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            # Process only persons (class 0) above the confidence threshold
            if int(cls) == 0 and conf >= CONFIDENCE_THRESHOLD:
                # Convert tensor values to float
                x1, y1, x2, y2 = [float(val) for val in box]
                conf_val = float(conf)
                # Draw bounding box for visualization
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 20, 147), 2)
                # Append detection as the format DeepSORT expects: ([x1, y1, x2, y2], confidence, class)
                detections.append([[x1, y1, x2, y2], conf_val, 0])
                logging.info(f"[DETECTION] Person detected at: {([x1, y1, x2, y2])} with confidence {conf_val:.2f}")
    return detections


def update_tracks_and_draw(frame, detections, now, punching_line, crossing_line, user_tracking):
    """
    Updates tracks using DeepSORT, draws track IDs on the frame,
    and checks for punching and crossing events.

    Each new track is initialized with a default color (magenta).
    When a punching event is detected, the color is updated to orange;
    when a crossing event is detected, it is updated to spring green.
    The stored color is reused on subsequent frames.
    """
    tracks = tracker.update_tracks(detections, frame=frame)
    logging.info(f"[INFO] Number of active tracks: {len(tracks)}")

    for track in tracks:
        track_id = track.track_id

        # Initialize track in user_tracking if new, with a default color (magenta)
        if track_id not in user_tracking:
            logging.warning(f"track_id: {track_id} is new; initializing in user tracking.")
            user_tracking[track_id] = {"punched": None, "crossed": None, "color": (255, 0, 255)}  # default magenta

        # Skip unconfirmed tracks
        if not track.is_confirmed():
            logging.warning(f"track_id: {track_id} is not confirmed, skipping.")
            continue

        if hasattr(track, "time_since_update") and track.time_since_update > 5:
            if track_id in user_tracking:
                logging.info(
                    f"track_id: {track_id} has not been updated for {track.time_since_update} frames; removing.")
                del user_tracking[track_id]
            continue

        # If the track has already crossed, just draw its ID with the stored color and skip further processing
        if user_tracking[track_id]["crossed"] is not None:
            id_color = user_tracking[track_id]["color"]
            logging.info(f"track_id: {track_id} already crossed; using stored color.")
            x1, y1, x2, y2 = track.to_tlbr()
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)
            continue

        # Get bounding box coordinates and compute center
        x1, y1, x2, y2 = track.to_tlbr()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Define key points on the bounding box (four corners and the center)
        bbox_points = [
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2),
            (center_x, center_y)
        ]

        # Check distances from these points to the punching and crossing lines
        punched = any(point_line_distance(pt, punching_line) < LINE_THRESHOLD for pt in bbox_points)
        crossed = any(point_line_distance(pt, crossing_line) < LINE_THRESHOLD for pt in bbox_points)

        logging.info(f"track_id: {track_id} \t punched: {punched} \t crossed: {crossed}")

        # Start with the stored color for consistency
        id_color = user_tracking[track_id]["color"]

        if crossed:
            id_color = (0, 255, 127)  # Spring green for crossing
            logging.info(f"track_id: {track_id} is near the crossing line.")
            if user_tracking[track_id]["crossed"] is None:
                user_tracking[track_id]["crossed"] = now  # Record crossing time
                if user_tracking[track_id]["punched"] is None:
                    user_tracking[track_id]["punched"] = now  # If missing, set punched time to now
                logging.info(f"[CROSS] User {track_id} successfully crossed at {now}")
        elif punched:
            id_color = (0, 165, 255)  # Orange for punching
            logging.info(f"track_id: {track_id} is near the punching line.")
            if user_tracking[track_id]["punched"] is None:
                user_tracking[track_id]["punched"] = now  # Record punching time
                logging.info(f"[PUNCH] User {track_id} punched at {now}")

        # Update the stored color so that the same color is used in future frames
        user_tracking[track_id]["color"] = id_color
        logging.info(f"track_id: {track_id} updated with color: {id_color}")

        # Draw the track ID on the frame using the determined color
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
                del user_tracking[track_id]
