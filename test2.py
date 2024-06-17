import cv2
from flask import Flask, Response
import mediapipe as mp
import numpy as np
import threading
import time
from picamera2 import Picamera2
import meshtastic
import meshtastic.serial_interface

app = Flask(__name__)

# Initialize Meshtastic interface
interface = meshtastic.serial_interface.SerialInterface()
interface.sendText("Node started")

# Initialize camera
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

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Global variables
frame = None
lock = threading.Lock()

# Eye aspect ratio thresholds and counters
EAR_THRESH = 0.25
CLOSED_EYE_FRAME_THRESH = 3  # In seconds
YAWN_THRESHOLD = 0.05  # You may need to adjust this threshold based on your observations
start_closed_eye_time = None
eye_closed_alert_sent = False
yawn_alert_sent = False

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global frame
    while True:
        time.sleep(0.1)  # Adjust this to control the frame rate
        with lock:
            if frame is None:
                continue
            _, encoded_image = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to send alert message in a separate thread
def send_alert_message(message):
    interface.sendText(message)

def calculate_lip_distance(face_landmarks):
    upper_lip_point = face_landmarks[13]
    lower_lip_point = face_landmarks[14]
    lip_distance = lower_lip_point.y - upper_lip_point.y
    return lip_distance

def process_video():
    global frame
    global start_closed_eye_time
    global eye_closed_alert_sent
    global yawn_alert_sent

    while True:
        success = True
        image = camera.capture_array()
        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_of_interest = [33, 263, 1, 61, 291, 199]
                left_eye_landmarks = [362, 385, 387, 263, 373, 380]
                right_eye_landmarks = [33, 160, 158, 133, 153, 144]
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in landmarks_of_interest:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                
                left_eye = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in left_eye_landmarks]
                right_eye = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in right_eye_landmarks]

                left_ear = eye_aspect_ratio(np.array(left_eye))
                right_ear = eye_aspect_ratio(np.array(right_eye))
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESH:
                    if start_closed_eye_time is None:
                        start_closed_eye_time = time.time()
                    else:
                        elapsed_time = time.time() - start_closed_eye_time
                        if elapsed_time > CLOSED_EYE_FRAME_THRESH:
                            cv2.putText(image, "ALERT! Eyes Closed for 3 seconds", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            if not eye_closed_alert_sent:
                                threading.Thread(target=send_alert_message, args=("Alert!! Eyes closed",)).start()
                                eye_closed_alert_sent = True
                else:
                    start_closed_eye_time = None
                    eye_closed_alert_sent = False

                lip_distance = calculate_lip_distance(face_landmarks.landmark)
                if lip_distance > YAWN_THRESHOLD:
                    cv2.putText(image, "ALERT! Yawning", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    if not yawn_alert_sent:
                        threading.Thread(target=send_alert_message, args=("ALERT! Yawning",)).start()
                        yawn_alert_sent = True
                elif lip_distance <= YAWN_THRESHOLD:
                    yawn_alert_sent = False

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                cv2.line(image, p1, p2, (255, 0, 0), 3)

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

        with lock:
            frame = image.copy()

if __name__ == '__main__':
    # Start a thread to process video
    threading.Thread(target=process_video, daemon=True).start()

    # Start Flask app
    app.run(host='0.0.0.0', port=8000)

camera.stop()
interface.close()
cv2.destroyAllWindows()
