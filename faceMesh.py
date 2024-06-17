import cv2
import mediapipe as mp
import numpy as np
import time
from picamera2 import Picamera2
import meshtastic
import meshtastic.serial_interface

interface = meshtastic.serial_interface.SerialInterface()
interface.sendText("Node started")

camera = Picamera2()

camera.resolution = (640, 360)
camera.rotation = 180
modes = camera.sensor_modes
mode = modes[1]
print('mode selected: ', mode)
camera_config = camera.create_still_configuration(raw={'format': mode['unpacked']}, sensor={'output_size': mode['size'], 'bit_depth': mode['bit_depth']}, main={"size": (640, 480)})
camera.configure(camera_config)
print("cam initialized")
camera.start()

time.sleep(0.1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Eye aspect ratio thresholds and counters
EAR_THRESH = 0.25
CLOSED_EYE_FRAME_THRESH = 3  # In seconds
start_closed_eye_time = None

YAWN_THRESHOLD = 0.05  # You may need to adjust this threshold based on your observations

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_lip_distance(face_landmarks):
    upper_lip_point = face_landmarks[13]  # Upper lip midpoint
    lower_lip_point = face_landmarks[14]  # Lower lip midpoint
    lip_distance = lower_lip_point.y - upper_lip_point.y
    return lip_distance

while True:
    success = True
    image = camera.capture_array()
    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    cv2.imshow("Raw", image)
    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define indexes for the landmarks of interest
            landmarks_of_interest = [33, 263, 1, 61, 291, 199]
            left_eye_landmarks = [362, 385, 387, 263, 373, 380]
            right_eye_landmarks = [33, 160, 158, 133, 153, 144]
            
            # Get the coordinates of the landmarks
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in landmarks_of_interest:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])
            
            # Eye tracking
            left_eye = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in left_eye_landmarks]
            right_eye = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in right_eye_landmarks]

            # Calculate the eye aspect ratio (EAR) for both eyes
            left_ear = eye_aspect_ratio(np.array(left_eye))
            right_ear = eye_aspect_ratio(np.array(right_eye))

            # Average the EAR for both eyes
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESH:
                if start_closed_eye_time is None:
                    start_closed_eye_time = time.time()
                else:
                    elapsed_time = time.time() - start_closed_eye_time
                    if elapsed_time > CLOSED_EYE_FRAME_THRESH:
                        cv2.putText(image, "ALERT! Eyes Closed for 3 seconds", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                        interface.sendText("Alert!! Eyes closed")
            else:
                start_closed_eye_time = None

            # Calculate lip distance for yawn detection
            lip_distance = calculate_lip_distance(face_landmarks.landmark)

            # If lip distance exceeds the threshold, consider it a yawn
            if lip_distance > YAWN_THRESHOLD:
                cv2.putText(image, "ALERT! Yawning", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                interface.sendText("ALERT! Yawning")
            # Convert face_2d and face_3d to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head is tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
                # interface.sendText("Alert!! Drowsy")
            elif x > 10:
                text = "Looking Up"
                # interface.sendText("Alert!! Drowsy")
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            end = time.time()
            totalTime = end - start
            fps = 60

            cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    cv2.imshow('Head Pose, Eye Tracking, and Yawn Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

camera.stop()
interface.close()
cv2.destroyAllWindows()