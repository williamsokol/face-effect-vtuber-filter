import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# --- Configuration Constants ---
# NOTE: Ensure 'deeplabv3.tflite' is in the same directory as this script.
MODEL_PATH = 'selfie_segmenter.tflite' 
BG_COLOR = (0, 255, 0)  # Gray color for the segmented background
MASK_COLOR = (255, 255, 255) # White color for the foreground mask

# --- STEP 1 & 2: Create the ImageSegmenter object and Options ---
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True  # We need the mask for visual output
    )
    # Initialize the segmenter outside the loop
    segmenter = vision.ImageSegmenter.create_from_options(options)
except ValueError:
    print(f"Error: Model file not found at '{MODEL_PATH}'. Please ensure it is in the same directory.")
    exit()

# --- STEP 3: Initialize Video Capture ---
# Use cv2.CAP_DSHOW for stable webcam access on Windows systems
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Starting live video segmentation using {MODEL_PATH}. Press ESC to exit.")

# --- STEP 4 & 5: Process Video Frames ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 1. Convert BGR (OpenCV default) to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Create the MediaPipe Image object from the NumPy array
    # We use SRGB format for standard RGB NumPy array data
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 3. Perform Segmentation
    # Use the standard `segment` method for simplicity in live video
    segmentation_result = segmenter.segment(mp_image)

    # 4. Extract the mask and prepare the output image
    if segmentation_result.category_mask:
        category_mask = segmentation_result.category_mask
        
        # The mask is a single-channel array (H, W). We stack it to 3 channels (H, W, 3).
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        
        # Create solid color background and foreground images
        image_data = mp_image.numpy_view() # Get the RGB frame back as a numpy array
        
        # --- Foreground (Person) ---
        # The output where condition is True will show the original image
        fg_image = image_data 
        
        # --- Background (Replaced with solid color) ---
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        
        # Apply the mask: show fg_image where condition is True, otherwise show bg_image
        output_image = np.where(condition, bg_image, fg_image)
        
        # Convert the final RGB output back to BGR for OpenCV display
        display_frame = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    else:
        # If no segmentation result, display the raw frame
        display_frame = frame 

    # 5. Display the resulting frame
    cv2.imshow('MediaPipe Image Segmenter (Live)', cv2.flip(display_frame, 1) )

    # Break the loop if the ESC key (0x1B) is pressed
    if cv2.waitKey(5) & 0xFF == 27: 
        break

# --- STEP 6: Release resources ---
segmenter.close() # Close the segmenter resource
cap.release()
cv2.destroyAllWindows()