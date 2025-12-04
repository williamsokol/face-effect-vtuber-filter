import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import List, Mapping, Optional, Tuple, Union

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

# --- Helper Function for Drawing Landmarks ---
# This function is typically provided in MediaPipe examples
# and is needed because the new API doesn't integrate drawing utilities
# directly into the task results object like the old one did.
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

# --- FaceLandmarker Setup ---

# IMPORTANT: You must download the model file 
# 'face_landmarker_v2_with_blendshapes.task' and place it 
# in the same directory as this script, or update the path.
model_path = 'face_landmarker_v2_with_blendshapes.task' 
# You can often find this model on the official MediaPipe documentation site.

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5, # New parameter names
    min_tracking_confidence=0.5
)

# Create the FaceLandmarker object
landmarker = vision.FaceLandmarker.create_from_options(options)

# --- Static Image Mode (Re-engineered from old code) ---
# NOTE: The original code had IMAGE_FILES=[] so this will not run unless you add files.
IMAGE_FILES = [] # Add your image file paths here, e.g., ["image1.png", "image2.jpg"]

for idx, file in enumerate(IMAGE_FILES):
    # Load the input image using MediaPipe's Image object
    mp_image = mp.Image.create_from_file(file)

    # Detect face landmarks
    detection_result = landmarker.detect(mp_image)

    # Process and draw the detection result
    # mp_image.numpy_view() converts the MediaPipe Image to a NumPy array (RGB format)
    annotated_image_rgb = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

    # Convert back to BGR for OpenCV's imwrite
    annotated_image_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'/tmp/annotated_image{idx}.png', annotated_image_bgr)
    print(f"Annotated image saved to /tmp/annotated_image{idx}.png")
    
# --- Webcam Input Mode (Re-engineered from old code) ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, bgr_image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Detect face landmarks
    detection_result = landmarker.detect(mp_image)

    # Draw the face mesh annotations on the image.
    # We use the original RGB image buffer for drawing, then convert back to BGR for display.
    if detection_result.face_landmarks:
        annotated_image_rgb = draw_landmarks_on_image(rgb_image, detection_result)
        display_image = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
    else:
        # If no face is detected, just display the original frame
        display_image = bgr_image

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Landmarker', cv2.flip(display_image, 1))
    if cv2.waitKey(5) & 0xFF == 27: # ESC key to exit
      break

cap.release()
cv2.destroyAllWindows()