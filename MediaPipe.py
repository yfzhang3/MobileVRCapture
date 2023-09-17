import cv2
import mediapipe as mp
import matplotlib.pyplot as plt


CAM_INDEX = 1

# activate webcam
feed = cv2.VideoCapture(CAM_INDEX)

# holistic model
static_image_mode = False # track one person across video
model_complexity = 1 # range 0 - 2
smooth_landmarks = False # we can blend the poses in Unity ourselves
min_detection_confidence = 0.5 # prefer higher accuracy over quicker results
min_tracking_confidence = 0.5 # tracking confidence

# media pipe tools
mp_drawing, mp_holistic = mp.solutions.drawing_utils, mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

# plot
plt.autoscale(False)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


def display_tracked_image(image, results):
    """Source: Google Media Pipe Github Examples."""
    
    image.flags.writeable = True
    
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style()
    )
    
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style()
    )
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

with mp_holistic.Holistic( static_image_mode = static_image_mode, 
                            model_complexity = model_complexity, 
                            smooth_landmarks = smooth_landmarks, 
                            min_detection_confidence = min_detection_confidence,
                            min_tracking_confidence = min_tracking_confidence) as holistic:
    
    while feed.isOpened():
        successful, frame = feed.read()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        if not successful:
            print("Ignoring frame.")
            continue
    
        frame.flags.writeable = False # improve performance slightly
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_results = holistic.process(rgb_frame)

        display_tracked_image(frame, holistic_results)
        
        # visualize landmarks
        if holistic_results.pose_world_landmarks:
        
            ax.clear()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            for landmark in holistic_results.pose_world_landmarks.landmark:
                ax.scatter(landmark.x, landmark.y, landmark.z)
        
            plt.pause(0.01)


feed.release()

