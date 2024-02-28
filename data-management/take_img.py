import cv2

# Capture the video stream from the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error opening webcam!")
    exit()

# Capture a single frame
ret, frame = cap.read()

# Check if frame capture was successful
if not ret:
    print("Error capturing frame!")
    exit()

# Save the captured image with a timestamp (optional)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"webcam_image_{timestamp}.jpg"
cv2.imwrite(filename, frame)

print(f"Image saved successfully as: {filename}")

# Release the webcam capture
cap.release()
