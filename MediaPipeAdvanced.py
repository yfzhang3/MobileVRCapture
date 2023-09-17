from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import mediapipe as mp
import cv2

# webcam using open cv
CAM_INDEX = 1
feed = cv2.VideoCapture(CAM_INDEX)
fps = feed.get(cv2.CAP_PROP_FPS)
timestep = int(1000 / fps)

# pose model options
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = 'pose_landmarker_heavy.task'

def three_break_output(pose: list):
    """Create output of the form x1, y1, z1, x2, y2, z2, x3, y3,"""
    joints = [0, 20, 22, 16, 14, 12, 24, 26, 28, 32, 30, 29, 27, 31, 25, 23, 11, 13, 15, 21, 19]
    output = ""
    for j in joints:
        output += f'{pose[j].x},{pose[j].y},{pose[j].z},'
    output = output[:-1]
    return output

def call_back(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.pose_landmarks:
        pose = result.pose_landmarks[0]
        return three_break_output(pose)

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=call_back,
    num_poses = 1) 

# time steps
frame_timestamp_ms = 0

with PoseLandmarker.create_from_options(options) as landmarker:
  
  while feed.isOpened():
        successful, frame = feed.read()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        if not successful:
            print("Ignoring frame.")
            continue
                
        # convert image to usable format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # run detection async
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        # timestep
        frame_timestamp_ms += timestep
        