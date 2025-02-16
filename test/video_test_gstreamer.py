import os
import cv2
import threading

# Manually set GStreamer environment path (Ensures OpenCV finds GStreamer)
GST_PATH = "C:\\gstreamer\\1.0\\msvc_x86_64"
os.environ["GST_PLUGIN_PATH"] = os.path.join(GST_PATH, "lib", "gstreamer-1.0")
os.environ["GST_PLUGIN_SYSTEM_PATH"] = os.path.join(GST_PATH, "lib", "gstreamer-1.0")
os.environ["PATH"] = os.pathsep.join([os.environ["PATH"], os.path.join(GST_PATH, "bin")])

# Verify if GStreamer is set correctly
print("GStreamer Path Set:", os.environ["PATH"])

# List of RTSP camera URLs (Use the working URL)
CAMERA_URLS = [
    "rtsp://admin:admin123@192.168.68.124:554/rtsp/streaming?channel=01&subtype=0"  # Use the exact working RTSP link
]

# Function to construct a GStreamer pipeline that matches the command-line version
def get_gstreamer_pipeline(rtsp_url):
    pipeline = (
        f"rtspsrc location={rtsp_url} latency=100 protocols=tcp ! "
        "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false"
    )
    return pipeline

# Function to process each RTSP stream
def process_stream(rtsp_url, window_name):
    gst_pipeline = get_gstreamer_pipeline(rtsp_url)
    print(f"Using GStreamer pipeline: {gst_pipeline}")  # Debugging output

    # Initialize OpenCV with GStreamer
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"Error: Unable to open stream {rtsp_url}. Check GStreamer installation and RTSP URL.")
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