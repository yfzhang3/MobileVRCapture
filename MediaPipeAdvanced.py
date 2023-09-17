import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
import cv2
import socket

# -- QUEUE -- 
image_queue = Queue()

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

# -- UNITY SERVER SOCKET --
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052) # 5052 random port that's unused


# -- UTILITY FUNCTIONS -- 
def draw_points(img: mp.Image, pose_landmarks, indices=[]):
    image = np.copy(img.numpy_view())
    if not indices:
        indices = list(range(len(pose_landmarks)))
    for j in indices:
        x, y = pose_landmarks[j].x, pose_landmarks[j].y
        x, y = round(x * img.width), round(y * img.height)
        image = cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_queue.put(image)
def geo_center(landmarks):
    xs, ys = [], []
    for lm in landmarks:
        xs.append(lm.x)
        ys.append(lm.y)
    return [sum(xs) / len(xs), sum(ys) / len(ys)]    
def transform_hand_to_pose(landmarks, a, b):
    pose_world_hand_landmarks = []
    for lm in landmarks:
        lm_new = []
        lm_new.append(a[0]*lm.x+b[0])
        lm_new.append(a[1]*lm.y+b[1])
        lm_new.append(a[2]*lm.z+b[2])
        pose_world_hand_landmarks.append(lm_new)
    return pose_world_hand_landmarks
def find_ab(z1, z2, z1_prime, z2_prime):
    # T: a * z1' + b = z1
    # T: a * z2' + b = z2
    # a (z1' - z2') = z1 - z2
    # b = z2 - a * z2'
    a = (z1 - z2) / (z1_prime - z2_prime)
    b = z2 - a * z2_prime
    return a, b
def crop_image(image: mp.Image, min_x, max_x, min_y, max_y):
    """Crop the media pipe image to the given rectangular bounds."""
    image_np = image.numpy_view()
    image_copy_np = np.copy(image_np)
    cropped_image_np = image_copy_np[min_y:max_y, min_x:max_x].astype(np.uint8)
    cropped_image = mp.Image(image_format=image.image_format, data=cropped_image_np)
    return cropped_image
def find_hand_sub_image(hand: list, image: mp.Image):
    """Return the image of the hand within the picture."""
    
    if any([hand[i].visibility < 0.2 for i in range(len(hand))]):
        return None
    
    w, h = image.width, image.height
    xc, yc = geo_center(hand)
    
    xcp, ycp = xc * w, yc * h
    buffer = max(w, h) // 4 # pixels
    
    min_x = int(max(xcp - buffer, 0))
    max_x = int(min(xcp + buffer, w))
    
    min_y = int(max(ycp - buffer, 0))
    max_y = int(min(ycp + buffer, h))
    
    w_crop = max_x - min_x
    h_crop = max_y - min_y
    
    if w_crop <= 0 or h_crop <= 0:
        return None
    
    sub_image = crop_image(image, min_x, max_x, min_y, max_y)

    return sub_image
def unity_landmarks(landmarks: list):
    """Create output of the form x1, y1, z1, x2, y2, z2, x3, y3,"""
    joints = [0, 20, 22, 16, 14, 12, 24, 26, 28, 32, 30, 29, 27, 31, 25, 23, 11, 13, 15, 21, 19]
    output = ""
    for j in joints:
        output += f'{landmarks[j].x},{landmarks[j].y},{landmarks[j].z},'
    output = output[:-1]
    return output
def unity_array(coords: list):
    output = ""
    for coord in coords:
        output += f'{coord[0]}, {coord[1]}, {coord[2]},'
    return output[:-1]
def pose_call_back(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """This function gets called when the pose model identifies a pose within a video frame. The hand model is triggered here."""
    if result.pose_world_landmarks and result.pose_landmarks:
        pose_world = result.pose_world_landmarks[0]
        pose_landmarks = result.pose_landmarks[0]
        pose_unity_format = unity_landmarks(pose_world)
        
        # draw the pose
        # joints = [0, 20, 22, 16, 14, 12, 24, 26, 28, 32, 30, 29, 27, 31, 25, 23, 11, 13, 15, 21, 19]
        # draw_points(output_image, pose_landmarks, joints)
        right_hand, left_hand = 'F', 'F'

        # wrist, thumb, index
        left_hand = [pose_landmarks[15], pose_landmarks[21], pose_landmarks[19]] 
        right_hand = [pose_landmarks[16], pose_landmarks[22], pose_landmarks[20]]
        
        left_hand_image = None # find_hand_sub_image(left_hand, output_image)
        right_hand_image = None # find_hand_sub_image(right_hand, output_image)
        
        # less hand markers
        subset = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]

        with HandLandmarker.create_from_options(hand_options) as landmarker:
            
            left_hand = 'F'
            right_hand = 'F'
            
            if left_hand_image:
                
                # detection algorithm on left hand
                hand_landmarker_left_result = landmarker.detect(left_hand_image)
                
                # view mp image of hand
                mp_np_image = left_hand_image.numpy_view()
                mp_np_image = cv2.cvtColor(mp_np_image, cv2.COLOR_RGB2BGR)
                
                # view hand landmarks
                if hand_landmarker_left_result.hand_landmarks:
                    # draw_points(mp.Image(image_format=output_image.image_format, data=mp_np_image), hand_landmarker_left_result.hand_landmarks[0])
                    # draw_points(mp.Image(image_format=output_image.image_format, data=mp_np_image), hand_landmarker_left_result.hand_landmarks[0])
                
                    hand_world_landmarks = hand_landmarker_left_result.hand_world_landmarks[0]
                    
                    # look at less data
                    hand_world_landmarks = [hand_world_landmarks[i] for i in subset]
                    
                    # wrist and thumb in pose world coordinates
                    wrist_left = np.array([pose_world[15].x, pose_world[15].y, pose_world[15].z])
                    thumb_left = np.array([pose_world[21].x, pose_world[21].y, pose_world[21].z])
                    
                    # wrist and thumb in hand world coordinates
                    wrist_left_prime = hand_world_landmarks[0]
                    wrist_left_prime = np.array([wrist_left_prime.x, wrist_left_prime.y, wrist_left_prime.z])
                    thumb_left_prime = hand_world_landmarks[4]
                    thumb_left_prime = np.array([thumb_left_prime.x, thumb_left_prime.y, thumb_left_prime.z])
                    
                    # transform parameters
                    a, b = find_ab(wrist_left, thumb_left, wrist_left_prime, thumb_left_prime)
                    pose_world_hand_coordinates = transform_hand_to_pose(hand_world_landmarks, a, b)

                    # final output
                    left_hand_unity_format = unity_array(pose_world_hand_coordinates)
                    pose_unity_format += left_hand_unity_format
                    left_hand = 'T'
                
            if right_hand_image:
                hand_landmarker_right_result = landmarker.detect(right_hand_image)
                
                if hand_landmarker_right_result.hand_landmarks:
                    # draw_points(mp.Image(image_format=output_image.image_format, data=mp_np_image), hand_landmarker_right_result.hand_landmarks[0])
                    # draw_points(mp.Image(image_format=output_image.image_format, data=mp_np_image), hand_landmarker_right_result.hand_landmarks[0])
                
                    hand_world_landmarks = hand_landmarker_right_result.hand_world_landmarks[0]
                    
                    # look at less data
                    hand_world_landmarks = [hand_world_landmarks[i] for i in subset]
                    
                    # wrist and thumb in pose world coordinates
                    wrist_right = np.array([pose_world[16].x, pose_world[16].y, pose_world[16].z])
                    thumb_right = np.array([pose_world[22].x, pose_world[22].y, pose_world[22].z])
                    
                    # wrist and thumb in hand world coordinates
                    wrist_right_prime = hand_world_landmarks[0]
                    wrist_right_prime = np.array([wrist_right_prime.x, wrist_right_prime.y, wrist_right_prime.z])
                    thumb_right_prime = hand_world_landmarks[4]
                    thumb_right_prime = np.array([thumb_right_prime.x, thumb_right_prime.y, thumb_right_prime.z])
                    
                    # transform parameters
                    a, b = find_ab(wrist_right, thumb_right, wrist_right_prime, thumb_right_prime)
                    pose_world_hand_coordinates = transform_hand_to_pose(hand_world_landmarks, a, b)
                    
                    # final output
                    right_hand_unity_format = unity_array(pose_world_hand_coordinates)
                    pose_unity_format += right_hand_unity_format
                    right_hand = 'T'

        sock.sendto(str.encode(left_hand + right_hand + pose_unity_format), serverAddressPort)


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
        
        cv2.imshow('Webcam', frame)

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
        
        # check image queue
        if not image_queue.empty():
            mp_np_image = image_queue.get()
            plt.imshow(mp_np_image)
            plt.axis('off')
            plt.show()
