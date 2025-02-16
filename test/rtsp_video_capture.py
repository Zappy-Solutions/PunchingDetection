import cv2

# Replace this with your RTSP URL
# rtsp_url = "rtsp://192.168.68.113:554/rtsp/streaming?channel=01&subtype=0"
rtsp_url = "rtsp://admin:525ForgetMe!@192.168.68.118:554/Streaming/Channels/101/"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()

# Retrieve properties from the stream
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Sometimes the stream might not provide FPS; use a default if needed.
if fps == 0:
    fps = 25

print(f"Stream properties: {width}x{height} at {fps} FPS")

# Define the codec and create a VideoWriter object.
# 'mp4v' is a common codec for MP4 files.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

print("Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received. Exiting...")
        break

    # Write the frame into the file 'output.mp4'
    out.write(frame)

    # Display the frame in a window
    cv2.imshow("RTSP Stream", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Recording stopped by user.")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()