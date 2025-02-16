# main.py
import cv2
import threading
import logging
from queue import Queue
from datetime import datetime

from config import HIKVISION_RTSP_URL, IMPACT_RTSP_URL, WINDOW_WIDTH, WINDOW_HEIGHT, FPS, MAX_QUEUE_SIZE, FRAME_SKIP
from logger_setup import setup_logging
from database import setup_database
from frame_reader import read_frames
from line_selection import select_line
from processing import process_detections, update_tracks_and_draw, check_violations
from violation_processor import process_violations

def main():
    setup_logging()
    conn, cursor = setup_database()

    # Select input mode: RTSP, webcam, or video file
    mode = input("Enter '1' for RTSP, '2' for webcam, or '3' for video file: ").strip()
    if mode == '1':
        camera_model = input("Enter '1' for IMPACT or '2' for HIKVISION: ").strip()
        if camera_model == '2':
            RTSP_URL = HIKVISION_RTSP_URL
            logging.info("Selected HIKVISION camera model")
        else:
            RTSP_URL = IMPACT_RTSP_URL
            logging.info("Selected IMPACT camera model")

        logging.info(f"Selected RTSP mode with URL: {RTSP_URL}")
        cap = cv2.VideoCapture(RTSP_URL)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"RTSP stream properties: {int(width)}x{int(height)} at {fps} FPS")

        # # Set the OpenCV capture properties
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
        # cap.set(cv2.CAP_PROP_FPS, FPS)
        # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # logging.info(f"RTSP stream properties: {int(width)}x{int(height)} at {fps} FPS")

    elif mode == '2':
        logging.info("Selected webcam mode")
        cap = cv2.VideoCapture(0)
    else:
        logging.info("Selected video file mode")
        video_file = input("Enter the path to the video file: ").strip()
        cap = cv2.VideoCapture(video_file)

    # Read an initial frame **before starting threads**
    ret, init_frame = cap.read()
    if not ret or init_frame is None:
        logging.error("[ERROR] Could not retrieve initial frame from the video source.")
        cap.release()
        return

    # Let the user select the punching and crossing lines before threading starts
    punching_line = select_line("Select Punching Line", init_frame.copy())
    crossing_line = select_line("Select Crossing Line", init_frame.copy())

    logging.info("Closing the windows")
    # **Clear previous frames by closing all OpenCV windows**
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensures windows are properly closed before proceeding

    # Initialize queues and tracking dictionaries
    frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    violation_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    user_tracking = {}       # Format: { track_id: {"punched": datetime, "crossed": bool} }
    violations_recorded = {} # Format: { track_id: date }

    # Start violation processing thread
    threading.Thread(target=process_violations, args=(violation_queue, cursor, conn), daemon=True).start()
    logging.info("[INFO] Violation processing thread started.")

    # Start frame reading thread
    threading.Thread(target=read_frames, args=(cap, frame_queue), daemon=True).start()

    frame_count = 0
    while True:
        frame = frame_queue.get()
        if frame is None:
            logging.info("[INFO] End of video stream. Exiting main loop...")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue  # Skip frames for performance

        # Draw the static lines
        cv2.line(frame, punching_line[0], punching_line[1], (0, 0, 255), 2)  # Red for punching
        cv2.line(frame, crossing_line[0], crossing_line[1], (0, 255, 0), 2)  # Green for crossing

        now = datetime.now()
        # logging.info(f"Before process_detections")
        detections = process_detections(frame)
        # logging.info(f"Before update_tracks_and_draw")
        update_tracks_and_draw(frame, detections, now, punching_line, crossing_line, user_tracking)
        # logging.info(f"Before check_violations")
        check_violations(frame, now, user_tracking, violations_recorded, violation_queue)
        # logging.info(f"After check_violations")

        logging.info(f"[QUEUE] Current Violation Queue Size: {violation_queue.qsize()}")
        cv2.namedWindow("Live CCTV Monitoring", cv2.WINDOW_NORMAL)
        cv2.imshow("Live CCTV Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("[INFO] 'q' pressed. Exiting main loop...")
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    logging.info("[INFO] Cleanup completed.")

if __name__ == '__main__':
    main()