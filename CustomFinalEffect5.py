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
ME_COLOR = (120/1.5, 120/1.5, 180/1.5)  # Pink color (R, G, B) for the bright person area
DARKNESS_THRESHOLD = 15     # Pixels below this brightness will be treated as the raw RGB area

# Brightness factors (pre-calculated)
BRIGHTNESS_FACTOR = 0.5     # how dark rgb is thresholded
BRIGHTNESS_FACTOR2 = 0.1    # how dark rgb is displayed

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

# Pre-create drawing specs outside loop (OPTIMIZATION)
eyebrow_style = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=1)
eye_style = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=10)

# Pre-allocate arrays (will be resized on first frame)
final_image_rgb = None
dark_rgb = None

# --- 4. Main Processing Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 1. Convert BGR (OpenCV) to RGB (MediaPipe) - single conversion
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Create grayscale version (for brightness check only)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # 3. Create the MediaPipe Image object for Segmentation
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 4. Perform Segmentation
    segmentation_result = segmenter.segment(mp_image)

    # 5. Apply THREE-WAY Masking Logic (Optimized for Speed)
    if segmentation_result.category_mask:
        category_mask = segmentation_result.category_mask
        
        # --- Darken grayscale for threshold comparison ---
        gray_frame = (gray_frame * BRIGHTNESS_FACTOR).astype(np.uint8)
        
        # --- Create Base Masks (vectorized operations) ---
        darkness_mask = gray_frame > DARKNESS_THRESHOLD
        mp_mask = category_mask.numpy_view() < 0.1
        
        # --- Define Three Regions ---
        bright_person_mask = np.logical_and(mp_mask, darkness_mask)
        dark_person_mask = np.logical_and(mp_mask, ~darkness_mask)  # Use ~ instead of np.logical_not
        
        # --- Pre-allocate final_image_rgb if needed ---
        if final_image_rgb is None or final_image_rgb.shape != rgb_frame.shape:
            final_image_rgb = np.empty_like(rgb_frame)
        
        # --- Fill with background color ---
        final_image_rgb[:] = BG_COLOR
        
        # --- Apply darkened RGB for dark person areas ---
        if dark_rgb is None or dark_rgb.shape != rgb_frame.shape:
            dark_rgb = np.empty_like(rgb_frame)
        
        # Darken in-place for speed
        np.multiply(rgb_frame, BRIGHTNESS_FACTOR2, out=dark_rgb, casting='unsafe')
        dark_rgb = dark_rgb.astype(np.uint8)
        
        final_image_rgb[dark_person_mask] = dark_rgb[dark_person_mask]
        
        # --- Apply ME_COLOR for bright person areas (direct assignment) ---
        final_image_rgb[bright_person_mask, 0] = ME_COLOR[0]
        final_image_rgb[bright_person_mask, 1] = ME_COLOR[1]
        final_image_rgb[bright_person_mask, 2] = ME_COLOR[2]
        
    else:
        final_image_rgb = rgb_frame.copy()

    # 6. Perform Face Tracking on RGB frame
    final_image_rgb.flags.writeable = False
    face_results = face_mesh.process(rgb_frame)
    final_image_rgb.flags.writeable = True

    # 7. Draw face landmarks (if detected)
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]  # Only 1 face
        
        # Draw eyes (combined to reduce function calls)
        mp_drawing.draw_landmarks(
            image=final_image_rgb,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=eye_style)
        
        mp_drawing.draw_landmarks(
            image=final_image_rgb,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=eye_style)
        
        # Draw eyebrows
        mp_drawing.draw_landmarks(
            image=final_image_rgb,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=eyebrow_style)
        
        mp_drawing.draw_landmarks(
            image=final_image_rgb,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=eyebrow_style)
        
        # Draw irises
        mp_drawing.draw_landmarks(
            image=final_image_rgb,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=eye_style)

    # 8. Convert RGB to BGR for display
    display_frame = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)

    # 9. Display the frame
    cv2.imshow('Face Tracking on Green Screen', cv2.flip(display_frame, 1))

    # Break on ESC key
    if cv2.waitKey(5) & 0xFF == 27: 
        break

# --- 5. Release resources ---
cap.release()
face_mesh.close()
segmenter.close()
cv2.destroyAllWindows()