import cv2

# Define the RTSP URL (Ensure the credentials are correct)
# Hikvision rtsp_url = "rtsp://admin:525ForgetMe!@192.168.68.118:554/Streaming/Channels/101/"

# Impact 4Mbps
rtsp_url = "rtsp://admin:admin123@192.168.68.124:554/rtsp/streaming?channel=01&subtype=0"


# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Define the output file details
output_filename = "hikvision_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
fps = 30  # Frames per second
window_width = 1280
window_height = 720

# Initialize the video writer
out = cv2.VideoWriter(output_filename, fourcc, fps, (window_width, window_height))

# Read and display video frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Resize the frame to fit the 720p window size
    frame_resized = cv2.resize(frame, (window_width, window_height))

    # Write the frame to the MP4 file
    out.write(frame_resized)

    # Display the resized frame
    cv2.imshow("Hikvision Camera Feed (720p)", frame_resized)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save the file
cv2.destroyAllWindows()

print(f"Video saved as {output_filename}")