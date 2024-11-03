import cv2
import sys
from deepface import DeepFace

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU and forces CPU usage

# Path to your video
video_path = "/external2/jessie/avspeech/test_data/f6PvAtLKerI_8.mp4"
# Desired time interval in milliseconds
desired_interval_ms = 80  # 80 ms

# Available backends and alignment options
backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]
alignment_modes = [True, False]

# User inputs for functionality, backend, and alignment mode
functionality = sys.argv[1]  # 'analyze', 'verify', 'find', 'represent', or 'extract_faces'
backend = sys.argv[2] if len(sys.argv) > 2 else backends[0]
align = alignment_modes[0] if len(sys.argv) < 4 or sys.argv[3].lower() == 'true' else alignment_modes[1]

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get total frames and frame rate
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
video_length = total_frames / frame_rate  # Length in seconds

# Calculate frame interval for 80 ms interval
frame_interval = int((desired_interval_ms / 1000) * frame_rate)

frame_count = 0
results = []

# Function to perform DeepFace operations
def deepface_operation(frame, functionality, backend, align):
    if functionality == 'verify':
        return DeepFace.verify(img1_path=frame, img2_path=frame, detector_backend=backend, align=align)
    elif functionality == 'find':
        return DeepFace.find(img_path=frame, db_path="my_db", detector_backend=backend, align=align)
    elif functionality == 'represent':
        return DeepFace.represent(img_path=frame, detector_backend=backend, align=align)
    elif functionality == 'extract_faces':
        return DeepFace.extract_faces(img_path=frame, detector_backend=backend, align=align)
    else:  # Default to 'analyze'
        return DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'], detector_backend=backend, align=align)

# Process the video frames
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video
    
    # Only analyze every calculated frame interval
    if frame_count % frame_interval == 0:
        try:
            # Perform the selected DeepFace operation
            analysis = deepface_operation(frame, functionality, backend, align)
            results.append({'frame': frame_count, 'analysis': analysis})
            print(f"Frame {frame_count}: {analysis}")

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

    frame_count += 1

# Release the video capture
cap.release()

print("Completed analysis for selected frames in the video.")
