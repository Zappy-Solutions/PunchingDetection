# frame_reader.py
import cv2
import logging
from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS

def read_frames(cap, queue):
    """
    Continuously reads frames from the video capture and puts them into the queue.
    When the stream ends, a sentinel (None) is placed in the queue.
    Resets the queue at the start.
    """
    logging.info("read_frames entry")

    # Reset the queue
    with queue.mutex:
        queue.queue.clear()
        queue.all_tasks_done.notify_all()
        queue.unfinished_tasks = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Optionally, you can resize the frame:
        # frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        queue.put(frame)
    queue.put(None)
    logging.info("[INFO] Frame reader thread ended.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Optionally, you can resize the frame:
        # frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        queue.put(frame)
    queue.put(None)
    logging.info("[INFO] Frame reader thread ended.")