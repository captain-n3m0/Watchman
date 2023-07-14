import cv2
import threading
import numpy as np

# Load pre-trained models for object detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load pre-trained model for pedestrian detection
pedestrian_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Initialize variables for analytics
face_count = 0
pedestrian_count = 0

# Function to perform face detection on a frame
def detect_faces(frame):
    global face_count
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face_count = len(faces)
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('CCTV Analysis', frame)

# Function to perform pedestrian detection on a frame
def detect_pedestrians(frame):
    global pedestrian_count
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)
    pedestrian_net.setInput(blob)
    detections = pedestrian_net.forward()

    # Draw bounding boxes around detected pedestrians
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            pedestrian_count += 1
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(frame, f"Pedestrians: {pedestrian_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('CCTV Analysis', frame)

# Function to capture and process video frames
def process_video_capture():
    video_capture = cv2.VideoCapture(0)  # Replace '0' with the video file path if you want to analyze a saved video
    while True:
        ret, frame = video_capture.read()
        detect_faces(frame)
        detect_pedestrians(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

# Start video processing thread
video_thread = threading.Thread(target=process_video_capture)
video_thread.start()
