# line_selection.py
import cv2
import logging

def select_line(window_name, frame):
    """
    Allows the user to select a horizontal line.
    The user clicks two points; the y-coordinate of the first click is used to draw a full-width line.
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
            # Force a full-width horizontal line based on the first click's y-coordinate
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