import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- MediaPipe Imports and Definitions ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# --- Configuration Constants for Green Screen ---
# IMPORTANT: Ensure 'selfie_segmenter.tflite' is in the same directory.
SEGMENTER_MODEL_PATH = 'selfie_segmenter.tflite' 
BG_COLOR = (0, 255, 0)      # Green color (R, G, B) for the background
ME_COLOR = (120, 120, 180)  # Pink color (R, G, B) for the bright person area
DARKNESS_THRESHOLD = 30     # Pixels below this brightness will be treated as the raw RGB area

# --- 1. Initialize Image Segmenter (Tasks API) ---
try:
    segmenter_base_options = python.BaseOptions(model_asset_path=SEGMENTER_MODEL_PATH)
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=segmenter_base_options,
        output_category_mask=True  
    )
    # Initialize the segmenter
    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
except ValueError:
    print(f"Error: Segmenter model file not found at '{SEGMENTER_MODEL_PATH}'.")
    exit()

# --- 2. Initialize Face Mesh (Legacy API) ---
# Use the FaceMesh context manager for initialization
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# --- 3. Initialize Video Capture ---
# Use cv2.CAP_DSHOW for stable webcam access on Windows systems
cap = cv2.VideoCapture(0) 

# Optional: Set a lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    segmenter.close()
    exit()

print("Starting live video feed with custom Green Screen and Face Tracking. Press ESC to exit.")

# --- 4. Main Processing Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 1. Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a grayscale version to check for brightness
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # 2. Create the MediaPipe Image object for Segmentation
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 3. Perform Segmentation
    segmentation_result = segmenter.segment(mp_image)

    # 4. Apply THREE-WAY Masking Logic (Optimized for Speed)
    final_image_rgb = rgb_frame.copy() # Default to raw frame
    
    if segmentation_result.category_mask:
        category_mask = segmentation_result.category_mask
        
        # --- Create Base Masks ---
        darkness_mask = gray_frame > DARKNESS_THRESHOLD # True = Bright, False = Dark
        mp_mask = category_mask.numpy_view() < 0.1       # True = Person, False = Background (FIXED: < instead of >)

        # --- Define Three Regions ---
        
        # 1. BRIGHT PERSON (Foreground, will be PINK)
        bright_person_mask = np.logical_and(mp_mask, darkness_mask)
        
        # 2. DARK PERSON (Thresholded, will be RAW RGB)
        dark_person_mask = np.logical_and(mp_mask, np.logical_not(darkness_mask))
        
        # --- Apply Colors via Optimized Indexing ---
        
        # 1. Initialize the final image with the background color (GREEN)
        final_image_rgb = np.full(rgb_frame.shape, BG_COLOR, dtype=np.uint8)
        
        # 2. Apply RAW RGB data (Person but Dark)
        final_image_rgb[dark_person_mask] = rgb_frame[dark_person_mask]

        # 3. Apply PINK color (Person and Bright)
        # FIX: Assign each color channel separately
        final_image_rgb[bright_person_mask, 0] = ME_COLOR[0]  # R channel
        final_image_rgb[bright_person_mask, 1] = ME_COLOR[1]  # G channel
        final_image_rgb[bright_person_mask, 2] = ME_COLOR[2]  # B channel
        
    else:
        # If no segmentation result, ensure final_image_rgb is defined
        final_image_rgb = rgb_frame.copy() 

    # 5. Perform Face Tracking on the final image
    final_image_rgb.flags.writeable = False
    
    # Process the final_image_rgb for accurate landmark placement
    # NOTE: FaceMesh process call should be on the final image (final_image_rgb)
    # not the original (rgb_frame) as it was in your previous attempt.
    face_results = face_mesh.process(rgb_frame) 
    
    # Draw the face mesh annotations onto the same image
    final_image_rgb.flags.writeable = True

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            
            # # Draw Face Contours
            mp_drawing.draw_landmarks(
                image=final_image_rgb,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            
            # Draw Irises
            mp_drawing.draw_landmarks(
                image=final_image_rgb,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

    # 6. Convert the final RGB output back to BGR for OpenCV display
    display_frame = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)

    # 7. Display the resulting frame
    cv2.imshow('Face Tracking on Green Screen', cv2.flip(display_frame, 1) )

    # Break the loop if the ESC key is pressed
    if cv2.waitKey(5) & 0xFF == 27: 
        break

# --- 5. Release resources ---
cap.release()
face_mesh.close()
segmenter.close()
cv2.destroyAllWindows()