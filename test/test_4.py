import os
import cv2
import threading
import logging
import time
import shutil  # For moving files
from datetime import datetime


# -------------------------------
# Logging Setup
# -------------------------------
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = 'rtsp_processing.log'
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
RTSP_URL = "rtsp://192.168.68.113:554/rtsp/streaming?channel=01&subtype=0"

RECORD_DURATION = 60  # Time in seconds to save RTSP stream before processing
OUTPUT_DIR = "recordings"
PROCESSED_DIR = "processed_files"

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Global flags for thread synchronization
stop_recording = threading.Event()
stop_processing = threading.Event()
current_video_file = None
file_lock = threading.Lock()


# -------------------------------
# RTSP Recording Thread
# -------------------------------
def record_rtsp_stream():
    """Records RTSP stream into rotating 1-minute video chunks."""
    global current_video_file

    while not stop_recording.is_set():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file = os.path.join(OUTPUT_DIR, f"recorded_{timestamp}.mp4")

        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            logging.error("[ERROR] Unable to open RTSP stream.")
            time.sleep(5)  # Wait before retrying
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Use 25 FPS if undefined

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

        start_time = time.time()
        logging.info(f"[INFO] Recording started: {video_file}")

        while time.time() - start_time < RECORD_DURATION:
            ret, frame = cap.read()
            if not ret:
                logging.warning("[WARNING] Frame read error! Possible packet loss.")
                continue

            # Handle White Frames
            if frame.mean() > 250:
                logging.warning("[WARNING] White patch detected! Skipping frame.")
                continue

            out.write(frame)  # Write frame to file

            if stop_recording.is_set():
                break

        cap.release()
        out.release()

        # Update the current processing file
        with file_lock:
            current_video_file = video_file

        logging.info(f"[INFO] Recording finished: {video_file}")


# -------------------------------
# Video Processing Thread
# -------------------------------
def process_video_file():
    """Processes recorded video files and moves them to 'processed_files' instead of deleting them."""
    global current_video_file

    while not stop_processing.is_set():
        with file_lock:
            if current_video_file is None:
                time.sleep(2)  # Wait if no file is available
                continue
            video_file = current_video_file
            current_video_file = None  # Reset for the next file

        if not os.path.exists(video_file):
            logging.error(f"[ERROR] Video file not found: {video_file}")
            continue

        logging.info(f"[INFO] Processing started: {video_file}")

        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video file

            # Draw some overlays (example: lines)
            cv2.line(frame, (0, 100), (frame.shape[1], 100), (0, 255, 0), 2)  # Example static line
            cv2.imshow("Processing Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("[INFO] Processing stopped by user.")
                stop_processing.set()
                break

        cap.release()
        cv2.destroyAllWindows()

        # Move processed file to processed_files directory
        processed_path = os.path.join(PROCESSED_DIR, os.path.basename(video_file))
        try:
            shutil.move(video_file, processed_path)
            logging.info(f"[INFO] Processed file moved to: {processed_path}")
        except Exception as e:
            logging.error(f"[ERROR] Failed to move file {video_file} -> {processed_path}: {str(e)}")


# -------------------------------
# Start Threads
# -------------------------------
recording_thread = threading.Thread(target=record_rtsp_stream, daemon=True)
processing_thread = threading.Thread(target=process_video_file, daemon=True)

recording_thread.start()
processing_thread.start()

# -------------------------------
# User Exit Handling
# -------------------------------
try:
    while True:
        time.sleep(1)  # Keep main thread alive
except KeyboardInterrupt:
    logging.info("[INFO] Stopping processes...")
    stop_recording.set()
    stop_processing.set()
    recording_thread.join()
    processing_thread.join()
    logging.info("[INFO] Program terminated.")