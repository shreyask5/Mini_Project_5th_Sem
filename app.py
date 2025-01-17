from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
import numpy as np
import mediapipe as mp
import time
import torch  # For YOLO

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for EAR and drowsiness detection
EAR_THRESHOLD = 0.19
DROWSY_FRAMES = 10

# Constants for head pose detection
PITCH_THRESHOLD = 10   # Threshold for pitch angle (in degrees)
HEAD_POSE_TIMER = 2  # Time in seconds to detect head position

# Constants for yawning detection
MAR_THRESHOLD = 0.40  # Threshold for MAR (above this value, yawning is detected)
YAWN_FRAMES = 10      # Number of consecutive frames to consider yawning

# Global variables to store the current status and timers
current_status = "Active"
head_pose_start_time = None
yawn_frame_count = 0

# Load YOLOv5 model for phone detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model
model.classes = [67]  # Filter to detect only 'cell phone' (class ID 67 in COCO dataset)

# Function to calculate EAR
def compute_ear(eye_points):
    vertical_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

# Function to calculate MAR
def compute_mar(mouth_points):
    vertical_dist1 = np.linalg.norm(mouth_points[13] - mouth_points[19])  # Top to bottom
    vertical_dist2 = np.linalg.norm(mouth_points[14] - mouth_points[18])  # Top to bottom
    horizontal_dist = np.linalg.norm(mouth_points[12] - mouth_points[16])  # Left to right
    mar = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return mar

# Video feed generator
def generate_frames():
    global current_status, head_pose_start_time, yawn_frame_count

    cap = cv2.VideoCapture(0)  # Use the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from the camera.")
            break

        # Process frame for drowsiness, head pose, and yawning
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Get image dimensions
        img_h, img_w, _ = image.shape

        # Drowsiness detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Extract eye landmarks
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Extract mouth landmarks
            mouth = landmarks[48:68]
            mar = compute_mar(mouth)

            # Drowsiness detection
            if ear < EAR_THRESHOLD:
                current_status = "Driver Drowsy!"
            else:
                current_status = "Active"

            # Yawning detection
            if mar > MAR_THRESHOLD:
                yawn_frame_count += 1
                if yawn_frame_count > YAWN_FRAMES:
                    current_status = "Yawning!"
            else:
                yawn_frame_count = 0

            # Draw EAR and MAR on the frame (optional, if you want to keep it)
            cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Head pose estimation using MediaPipe
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract key landmarks for head pose estimation
                face_2d = []
                face_3d = []

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert to NumPy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w
                cam_matrix = np.array([
                    [focal_length, 0, img_h / 2],
                    [0, focal_length, img_w / 2],
                    [0, 0, 1]
                ])

                # The distortion matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP to get rotation and translation vectors
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the pitch angle (up/down)
                x = angles[0] * 360  # Pitch (up/down)

                # Head pose detection logic
                if x > PITCH_THRESHOLD:  # Head is facing upwards
                    if head_pose_start_time is None:
                        head_pose_start_time = time.time()
                    elif time.time() - head_pose_start_time > HEAD_POSE_TIMER:
                        current_status = "Head Up!"
                elif x < -PITCH_THRESHOLD:  # Head is facing downwards
                    if head_pose_start_time is None:
                        head_pose_start_time = time.time()
                    elif time.time() - head_pose_start_time > HEAD_POSE_TIMER:
                        current_status = "Head Down!"
                else:
                    head_pose_start_time = None
                    if current_status not in ["Driver Drowsy!", "Yawning!"]:
                        current_status = "Active"

        # Phone detection using YOLO
        yolo_results = model(image)  # Run YOLO inference
        detections = yolo_results.pandas().xyxy[0]  # Get detection results

        # Check if a phone is detected
        phone_detected = False
        for _, detection in detections.iterrows():
            if detection['name'] == 'cell phone' and detection['confidence'] > 0.5:  # Confidence threshold
                phone_detected = True
                # Draw bounding box around the phone
                cv2.rectangle(image, (int(detection['xmin']), int(detection['ymin'])),
                              (int(detection['xmax']), int(detection['ymax'])), (0, 255, 0), 2)
                cv2.putText(image, f"Phone: {detection['confidence']:.2f}",
                            (int(detection['xmin']), int(detection['ymin']) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update status if phone is detected
        if phone_detected:
            current_status = "Phone in Use!"

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Yield frame for video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get current status
@app.route('/status')
def status():
    return jsonify(status=current_status)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)