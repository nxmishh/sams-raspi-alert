import cv2
from flask import Flask, Response, jsonify
import mediapipe as mp
import numpy as np
import threading
import time
import meshtastic
import meshtastic.serial_interface
import csv
import os

app = Flask(__name__)

# Initialize Meshtastic interface
interface = meshtastic.serial_interface.SerialInterface()
interface.sendText("Node started")

# Initialize camera
camera = cv2.VideoCapture(0)  # Use the first webcam available
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
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
YAWN_THRESHOLD = 0.05  # Need to be adjusted based on observations
LOOK_UP_DOWN_THRESH = 5  # In seconds
start_closed_eye_time = None
start_look_up_down_time = None
eye_closed_alert_sent = False
yawn_alert_sent = False
look_up_down_alert_sent = False

# CSV file setup
csv_file = 'alert_log.csv'
csv_columns = ['timestamp', 'alert_type', 'details']

if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

def log_event(alert_type, details):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writerow({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'alert_type': alert_type,
            'details': details
        })

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts', methods=['GET'])
def get_alerts():
    with open(csv_file, 'r') as file:
        csv_data = list(csv.DictReader(file))
    return jsonify(csv_data)

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
    global start_look_up_down_time
    global eye_closed_alert_sent
    global yawn_alert_sent
    global look_up_down_alert_sent

    while True:
        success, image = camera.read()
        if not success:
            continue
        
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
                                log_event("Eyes Closed", "Eyes closed for more than 3 seconds")
                                eye_closed_alert_sent = True
                else:
                    start_closed_eye_time = None
                    eye_closed_alert_sent = False

                lip_distance = calculate_lip_distance(face_landmarks.landmark)
                if lip_distance > YAWN_THRESHOLD:
                    cv2.putText(image, "ALERT! Yawning", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    if not yawn_alert_sent:
                        threading.Thread(target=send_alert_message, args=("ALERT! Yawning",)).start()
                        log_event("Yawning", "Yawning detected")
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
                    start_look_up_down_time = None
                elif y > 10:
                    text = "Looking Right"
                    start_look_up_down_time = None
                elif x < -10:
                    text = "Looking Down"
                    if start_look_up_down_time is None:
                        start_look_up_down_time = time.time()
                    else:
                        elapsed_time = time.time() - start_look_up_down_time
                        if elapsed_time > LOOK_UP_DOWN_THRESH:
                            cv2.putText(image, "ALERT! Looking Down", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            if not look_up_down_alert_sent:
                                threading.Thread(target=send_alert_message, args=("ALERT! Looking Down",)).start()
                                log_event("Looking Down", "Looking down for more than 5 seconds")
                                look_up_down_alert_sent = True
                elif x > 10:
                    text = "Looking Up"
                    if start_look_up_down_time is None:
                        start_look_up_down_time = time.time()
                    else:
                        elapsed_time = time.time() - start_look_up_down_time
                        if elapsed_time > LOOK_UP_DOWN_THRESH:
                            cv2.putText(image, "ALERT! Looking Up", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                            if not look_up_down_alert_sent:
                                threading.Thread(target=send_alert_message, args=("ALERT! Looking Up",)).start()
                                log_event("Looking Up", "Looking up for more than 5 seconds")
                                look_up_down_alert_sent = True
                else:
                    text = "Forward"
                    start_look_up_down_time = None
                    look_up_down_alert_sent = False

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

camera.release()
interface.close()
cv2.destroyAllWindows()
