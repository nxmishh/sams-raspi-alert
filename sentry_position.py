import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

def is_body_fully_visible(landmarks):
    essential_landmarks = [
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE
    ]
    for landmark in essential_landmarks:
        if landmarks[landmark.value].visibility < 0.5 or \
           landmarks[landmark.value].y < 0.0 or landmarks[landmark.value].y > 1.0:
            return False
    return True

def check_sitting_or_standing(landmarks):
    # Get the y-coordinates of the hips and knees
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

    # Calculate the average y-coordinates
    avg_hip_y = (left_hip_y + right_hip_y) / 2
    avg_knee_y = (left_knee_y + right_knee_y) / 2

    # Determine if the person is standing or sitting based on threshold
    if avg_hip_y < avg_knee_y * 0.8:
        return "Standing"
    else:
        return "Sitting"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks and is_body_fully_visible(results.pose_landmarks.landmark):
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        status = check_sitting_or_standing(results.pose_landmarks.landmark)
        cv2.putText(image, f'Status: {status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'Status: No Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Pose', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
