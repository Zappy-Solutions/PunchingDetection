import cv2
import threading

# List of RTSP camera URLs (Replace with your camera IPs and credentials)
CAMERA_URLS = [
    "rtsp://admin:admin123@192.168.29.40:554/rtsp/streaming?channel=01&subtype=0"
    #, "rtsp://username:password@camera_ip2:port/stream"
]

# Function to construct GStreamer pipeline
def get_gstreamer_pipeline(rtsp_url):
    pipeline = (
        f"rtspsrc location={rtsp_url} latency=100 ! "
        "decodebin ! videoconvert ! appsink"
    )
    return pipeline

# Function to process each RTSP stream
def process_stream(rtsp_url, window_name):
    gst_pipeline = get_gstreamer_pipeline(rtsp_url)
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"Error: Unable to open stream {rtsp_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to fetch frame from {rtsp_url}")
            break

        # Display the video feed
        cv2.imshow(window_name, frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Launch multiple RTSP streams in parallel threads
threads = []
for i, url in enumerate(CAMERA_URLS):
    t = threading.Thread(target=process_stream, args=(url, f"Camera {i+1}"))
    t.start()
    threads.append(t)

# Wait for all threads to finish
for t in threads:
    t.join()
