import cv2

ip_addr = "192.168.68.115"
# Construct the RTSP URL from the IP address
rtsp_url = f"rtsp://admin:admin123@{ip_addr}:554/rtsp/streaming?channel=01&subtype=0"
# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Define the output file details
output_filename = "Impact_4MP_hikvision_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
fps = 15  # Frames per second
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
    cv2.imshow("4MP Impact Camera Feed (720p)", frame_resized)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save the file
cv2.destroyAllWindows()

print(f"Video saved as {output_filename}")