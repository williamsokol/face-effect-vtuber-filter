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
BG_COLOR = (0, 255, 0)  # Green color (R, G, B) for the new background
ME_COLOR = (128, 50, 128)  # Green color (R, G, B) for the new background


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

print("Starting live video feed with Green Screen and Face Tracking. Press ESC to exit.")

# --- 4. Main Processing Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 1. Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Create the MediaPipe Image object for Segmentation
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 3. Perform Segmentation
    segmentation_result = segmenter.segment(mp_image)

    # 4. Apply Green Screen Effect
    final_image_rgb = rgb_frame # Start with the original RGB frame
    
    if segmentation_result.category_mask:
        category_mask = segmentation_result.category_mask
        
        # Stack the single-channel mask to 3 channels (H, W, 3) for the np.where function
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        
        # Create solid color background
        bg_image = np.zeros(rgb_frame.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        me_image = np.zeros(rgb_frame.shape, dtype=np.uint8)
        me_image[:] = ME_COLOR
        
        # Apply the mask: show original frame where condition is True (person), otherwise show bg_image (green screen)
        # NOTE: The segmentation model returns a mask for the *person*.
        # np.where(condition, a, b) selects elements from 'a' where 'condition' is True, and 'b' otherwise.
        # We want the *background* to be green, so we swap the foreground/background usage based on the mask.
        final_image_rgb = np.where(condition, bg_image, me_image)

    # 5. Perform Face Tracking on the Green-Screened Image
    # NOTE: The legacy Face Mesh requires the image to be marked as not writeable for processing
    final_image_rgb.flags.writeable = False
    face_results = face_mesh.process(rgb_frame)
    
    # Draw the face mesh annotations onto the same image
    final_image_rgb.flags.writeable = True

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            
            # Draw Face Contours
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
            
            # Draw TESSELATION
            mp_drawing.draw_landmarks(
              image=final_image_rgb,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
            # Note: The dense FACEMESH_TESSELATION is intentionally omitted here 
            # as it is a major source of CPU bottleneck/lag.

    # 6. Convert the final RGB output back to BGR for OpenCV display
    display_frame = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)

    # 7. Display the resulting frame
    cv2.imshow('Face Tracking on Green Screen', cv2.flip(display_frame, 1) )

    # Break the loop if the ESC key is pressed
    if cv2.waitKey(5) & 0xFF == 27: 
        break

# --- 5. Release resources ---
cap.release()
face_mesh.close() # Close the legacy FaceMesh resource
segmenter.close() # Close the segmenter resource
cv2.destroyAllWindows()