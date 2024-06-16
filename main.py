import cv2
import threading
import numpy as np

# Load pre-trained models for face detection and pedestrian detection using SSD
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
pedestrian_net = cv2.dnn.readNetFromCaffe('pedestrian_deploy.prototxt', 'pedestrian_iter_120000.caffemodel')

# Initialize variables for analytics
face_count = 0
pedestrian_count = 0

# Function to perform face detection on a frame
def detect_faces(frame):
    global face_count
    face_count = 0
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Function to perform pedestrian detection on a frame
def detect_pedestrians(frame):
    global pedestrian_count
    pedestrian_count = 0
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    pedestrian_net.setInput(blob)
    detections = pedestrian_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            pedestrian_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(frame, f"Pedestrians: {pedestrian_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Function to capture and process video frames
def process_video_capture():
    video_capture = cv2.VideoCapture(0)  # Replace '0' with the video file path if you want to analyze a saved video
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        detect_faces(frame)
        detect_pedestrians(frame)
        cv2.imshow('CCTV Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

# Start video processing thread
video_thread = threading.Thread(target=process_video_capture)
video_thread.start()
