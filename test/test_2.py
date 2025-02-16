import os
import cv2
import threading
import logging
import numpy as np
from queue import Queue
from datetime import datetime


# -------------------------------
# Logging Setup
# -------------------------------
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
FRAME_SKIP = 5  # Process every Nth frame for performance
cv2.setUseOptimized(True)

# -------------------------------
# Input Mode: RTSP, Webcam, or Video File
# -------------------------------
rtsp_url = "rtsp://192.168.68.113:554/rtsp/streaming?channel=01&subtype=0"

mode = input("Enter '1' for RTSP, '2' for webcam, or '3' for video file: ").strip()
if mode == '1':
    # Retrieve webcam properties for reference
    temp_webcam = cv2.VideoCapture(0)
    if not temp_webcam.isOpened():
        logging.error("Webcam not available to read properties. Using default settings.")
        width, height, fps = 640, 480, 10
    else:
        width = int(temp_webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = temp_webcam.get(cv2.CAP_PROP_FPS) or 10
        temp_webcam.release()

    logging.info(f"Webcam properties: {width}x{height} at {fps} FPS")

    logging.info(f"Selected RTSP mode with URL: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
elif mode == '2':
    logging.info("Selected webcam mode")
    cap = cv2.VideoCapture(0)
else:
    logging.info("Selected video file mode")
    VIDEO_FILE = input("Enter the path to the video file: ").strip()
    cap = cv2.VideoCapture(VIDEO_FILE)

# -------------------------------
# Frame Reading Thread
# -------------------------------
frame_queue = Queue(maxsize=10)


def read_frames(cap, queue):
    """Continuously read frames, discarding blank ones, and enqueue them."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("[WARNING] Blank frame detected. Skipping...")
            continue  # Skip blank frames

        # Check if the frame is blank (mostly white/gray)
        if np.mean(frame) > 250:
            logging.warning("[WARNING] Mostly white frame detected. Skipping...")
            continue  # Skip frames with too much brightness

        queue.put(frame)  # Only enqueue valid frames

    queue.put(None)  # Sentinel value for termination
    logging.info("[INFO] Frame reader thread ended.")


threading.Thread(target=read_frames, args=(cap, frame_queue), daemon=True).start()


# -------------------------------
# Line Selection with Full Width
# -------------------------------
def select_line(window_name, frame):
    """
    Allows the user to select a full-width line by clicking two points.
    Automatically extends the line across the entire width of the frame.
    """
    points = []
    clone = frame.copy()
    height, width, _ = frame.shape

    def mouse_callback(event, x, y, flags, param):
        nonlocal clone, points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)  # Red dots for selection
            cv2.imshow(window_name, clone)

        if len(points) == 2:
            y_selected = points[0][1]  # Y-coordinate of the first click
            points = [(0, y_selected), (width, y_selected)]  # Full-width line

            clone = frame.copy()
            cv2.line(clone, points[0], points[1], (0, 255, 0), 2)  # Green line
            cv2.imshow(window_name, clone)

    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_callback)
    print(f"Click two points to set the {window_name}. Press 'c' to confirm.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(points) == 2:
            break

    cv2.setMouseCallback(window_name, lambda *args: None)  # Disable callback
    cv2.destroyWindow(window_name)
    logging.info(f"[INFO] Selected {window_name}: {points[0]} to {points[1]}")
    return points[0], points[1]


# Ensure an initial frame is available for selection
while True:
    init_frame = frame_queue.get()
    if init_frame is not None:
        break  # Proceed only when a valid frame is available

punching_line = select_line("Select Punching Line", init_frame.copy())
crossing_line = select_line("Select Crossing Line", init_frame.copy())


# -------------------------------
# Main Processing Loop
# -------------------------------
def main_loop():
    """Processes frames, discards blank ones, and displays the video feed."""
    while True:
        frame = frame_queue.get()
        if frame is None:
            logging.info("[INFO] End of video stream. Exiting main loop...")
            break

        # Draw the selected lines
        cv2.line(frame, punching_line[0], punching_line[1], (0, 0, 255), 2)  # Red for punching
        cv2.line(frame, crossing_line[0], crossing_line[1], (0, 255, 0), 2)  # Green for crossing

        # Resize frame before display to ensure proper visibility
        frame_resized = cv2.resize(frame, (800, 600))

        cv2.imshow("Live CCTV Monitoring", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("[INFO] 'q' pressed. Exiting main loop...")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("[INFO] Cleanup completed.")


# -------------------------------
# Run the Main Loop
# -------------------------------
if __name__ == '__main__':
    main_loop()
