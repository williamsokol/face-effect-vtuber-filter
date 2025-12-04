import cv2

# Initialize the video capture object
# 0 refers to the default webcam. Use 1, 2, etc., for other cameras.
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream (webcam).")
    exit()

print("--- Live Video Feed ---")
print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    # 'ret' (or 'success') is a boolean flag, 'frame' is the actual video frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    # The first argument is the window name
    cv2.imshow('LIVE WEBCAM FEED', frame)

    # Wait for a key press (5ms delay) and check if it's the 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()