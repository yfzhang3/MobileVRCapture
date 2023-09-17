import mediapipe as mp
import numpy as np
import math
import cv2
import socket

# -- WEBCAM -- 
CAM_INDEX = 0
feed = cv2.VideoCapture(CAM_INDEX)
fps = feed.get(cv2.CAP_PROP_FPS)
timestep = int(1000 / fps)
frame_timestamp_ms = 0

# -- GENERAL OPTIONS -- 
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# -- POSE MODEL OPTIONS -- 
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
pose_model_path = 'pose_landmarker_heavy.task'


# -- HAND MODEL OPTIONS -- 
hand_model_path = 'hand_landmarker.task'
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# -- SOCKET --
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052) # 5052 random port that's unused

# -- UTILITY FUNCTIONS -- 
def transform_hand_to_pose(landmarks, a, b):
    pose_world_hand_landmarks = []
    for lm in landmarks:
        lm_new = []
        lm_new.append(a[0]*lm.x+b[0])
        lm_new.append(a[1]*lm.y+b[1])
        lm_new.append(a[2]*lm.z+b[2])
        pose_world_hand_landmarks.append(lm_new)
    return pose_world_hand_landmarks
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    zs = [lm.z for lm in landmarks]
    return ([sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)])
def find_ab(z1, z2, z1_prime, z2_prime):
    # format z1: [x, y, z]
    a_values = []
    b_values = []
    for i in range(3):
        coord_z1 = z1[i]
        coord_z2 = z2[i]
        coord_z1_prime = z1_prime[i]
        coord_z2_prime = z2_prime[i]
        a = (coord_z2_prime - coord_z1_prime) / (coord_z2 - coord_z1)
        b = coord_z1_prime - a * coord_z1
        a_values.append(a)
        b_values.append(b)
    return a_values, b_values
def crop_image(image: mp.Image, min_x, max_x, min_y, max_y):
    """Crop the media pipe image to the given rectangular bounds."""
    image_np = image.numpy_view()
    image_copy_np = np.copy(image_np)
    cropped_image_np = image_copy_np[min_y:max_y, min_x:max_x].astype(np.uint8)
    cropped_image = mp.Image(image_format=image.image_format, data=cropped_image_np)
    return cropped_image
def find_hand_sub_image(hand: list, image: mp.Image):
    """Return the image of the hand within the picture."""
    w, h = image.width, image.height
    wrist, thumb, index = hand
    hand_cord_length = math.sqrt((wrist.x - index.x)**2 + (thumb.y - index.y)**2)
    buffer = hand_cord_length
    
    min_x = round(max(min(wrist.x, thumb.x, index.x) - buffer, 0) * w)
    max_x = round(min(max(wrist.x, thumb.x, index.x) + buffer, 1) * w)
    
    min_y = round(max(min(wrist.y, thumb.y, index.y) - buffer, 0) * h)
    max_y = round(min(max(wrist.y, thumb.y, index.y) + buffer, 1) * h)
    
    if (max_y < min_y or max_x < min_x):
        return None
    
    sub_image = crop_image(image, min_x, max_x, min_y, max_y)
    
    return sub_image
def unity_format_cords(pose: list):
    """Create output of the form x1, y1, z1, x2, y2, z2, x3, y3,"""
    joints = [0, 20, 22, 16, 14, 12, 24, 26, 28, 32, 30, 29, 27, 31, 25, 23, 11, 13, 15, 21, 19]
    output = ""
    for j in joints:
        output += f'{pose[j].x},{pose[j].y},{pose[j].z},'
    output = output[:-1]
    return output
def pose_call_back(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """This function gets called when the pose model identifies a pose within a video frame. The hand model is triggered here."""
    if result.pose_world_landmarks and result.pose_landmarks:
        pose_world = result.pose_world_landmarks[0]
        pose_landmarks = result.pose_landmarks[0]
        pose_unity_format = unity_format_cords(pose_world)
        sock.sendto(str.encode(pose_unity_format), serverAddressPort)
        
        # wrist, thumb, index
        left_hand = [pose_landmarks[15], pose_landmarks[21], pose_landmarks[19]] 
        right_hand = [pose_landmarks[16], pose_landmarks[22], pose_landmarks[20]]
        
        left_hand_image = find_hand_sub_image(left_hand, output_image)
        right_hand_image = find_hand_sub_image(right_hand, output_image)

        with HandLandmarker.create_from_options(hand_options) as landmarker:
            if left_hand_image:
                # detection algorithm on left hand
                hand_landmarker_left_result = landmarker.detect(left_hand_image)
                
                if hand_landmarker_left_result.hand_world_landmarks:
                    # wrist and thumb in pose world coordinates
                    wrist_left = [pose_world[15].x, pose_world[15].y, pose_world[15].z]
                    thumb_left = [pose_world[21].x, pose_world[21].y, pose_world[21].z]
                    
                    # wrist and thumb in hand world coordinates
                    wrist_left_prime = hand_landmarker_left_result.hand_world_landmarks[0]
                    wrist_left_prime = [wrist_left_prime.x, wrist_left_prime.y, wrist_left_prime.z]
                    thumb_left_prime = hand_landmarker_left_result.hand_world_landmarks[4]
                    thumb_left_prime = [thumb_left_prime.x, thumb_left_prime.y, thumb_left_prime.z]
                    
                    # transform parameters
                    a, b = find_ab(wrist_left, thumb_left, wrist_left_prime, thumb_left_prime)
                    pose_world_hand_coordinates = transform_hand_to_pose(hand_landmarker_left_result.pose_world)
                    
                    # final output
                    left_hand_unity_format = unity_format_cords(pose_world_hand_coordinates)
                    pose_unity_format.append(left_hand_unity_format)
                
            if right_hand_image:
                hand_landmarker_right_result = landmarker.detect(right_hand_image)

                if hand_landmarker_right_result.hand_world_landmarks:
                    # wrist and thumb in pose world coordinates
                    wrist_right = [pose_world[16].x, pose_world[16].y, pose_world[16].z]
                    thumb_right = [pose_world[22].x, pose_world[22].y, pose_world[22].z]
                    
                    # wrist and thumb in hand world coordinates
                    wrist_right_prime = hand_landmarker_right_result.hand_world_landmarks[0]
                    wrist_right_prime = [wrist_right_prime.x, wrist_right_prime.y, wrist_right_prime.z]
                    thumb_right_prime = hand_landmarker_right_result.hand_world_landmarks[4]
                    thumb_right_prime = [thumb_right_prime.x, thumb_right_prime.y, thumb_right_prime.z]
                    
                    # transform parameters
                    a, b = find_ab(wrist_right, thumb_right, wrist_right_prime, thumb_right_prime)
                    pose_world_hand_coordinates = transform_hand_to_pose(hand_landmarker_right_result.pose_world)
                    
                    # final output
                    right_hand_unity_format = unity_format_cords(pose_world_hand_coordinates)
                    pose_unity_format.append(right_hand_unity_format)

        print(pose_unity_format)
        return pose_unity_format

# -- MODEL OPTIONS -- 
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=pose_call_back,
    num_poses = 1) 
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands = 2)


# -- BEGIN MODEL -- 
with PoseLandmarker.create_from_options(pose_options) as landmarker:
  
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
     
        
# -- NOT USED --
def draw_points(frame, pose_landmarks):
    image = np.copy(frame.numpy_view())
    joints = [0, 20, 22, 16, 14, 12, 24, 26, 28, 32, 30, 29, 27, 31, 25, 23, 11, 13, 15, 21, 19]
    for j in joints:
        x, y = pose_landmarks[j].x, pose_landmarks[j].y
        x, y = round(x * frame.width), round(y * frame.height)
        image = cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
