import os
import pickle
import threading
import time

import cv2
import face_recognition
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

REGISTERED_FACES_PATH = "registered_faces.pkl"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAPTURE_FPS = 30
STREAM_FPS = 30
JPEG_QUALITY = 80
RECOGNITION_SCALE = 0.25
RECOGNITION_FPS = 5
MATCH_TOLERANCE = 0.5

app = Flask(__name__)

# Load registered faces from file
if os.path.exists(REGISTERED_FACES_PATH):
    with open(REGISTERED_FACES_PATH, "rb") as f:
        registered_faces = pickle.load(f)
else:
    registered_faces = {}

# Global variable to temporarily hold face encoding for new faces
temp_face_encoding = None
temp_face_lock = threading.Lock()
known_face_names = []
known_face_encodings = []
known_faces_lock = threading.Lock()


def refresh_known_face_cache():
    """Refresh cached names and encodings for faster vectorized matching."""
    global known_face_names, known_face_encodings
    with known_faces_lock:
        known_face_names = list(registered_faces.keys())
        known_face_encodings = [info["encoding"] for info in registered_faces.values()]


refresh_known_face_cache()


class VideoProcessor:
    """Capture webcam frames, recognize faces, and produce fast MJPEG frames.

    The browser receives a normal multipart MJPEG video stream. Capture,
    recognition, and streaming are separated so slow face encoding cannot block
    the live video feed.
    """

    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_lock = threading.Lock()
        self.detections_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.stopped = threading.Event()
        self.latest_frame = None
        self.latest_detections = []
        self.latest_status = "Face: None"

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)

    def start(self):
        self.capture_thread.start()
        self.recognition_thread.start()
        return self

    def _capture_loop(self):
        frame_delay = 1 / CAPTURE_FPS
        while not self.stopped.is_set():
            started_at = time.perf_counter()
            ret, frame = self.capture.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

            elapsed = time.perf_counter() - started_at
            time.sleep(max(0.001, frame_delay - elapsed))

    def _recognition_loop(self):
        recognition_delay = 1 / RECOGNITION_FPS
        while not self.stopped.is_set():
            started_at = time.perf_counter()
            frame = self.get_frame()
            if frame is not None:
                detections = recognize_faces(frame)
                with self.detections_lock:
                    self.latest_detections = detections
                self._update_status(detections)

            elapsed = time.perf_counter() - started_at
            time.sleep(max(0.001, recognition_delay - elapsed))

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_detections(self):
        with self.detections_lock:
            return list(self.latest_detections)

    def get_status(self):
        with self.status_lock:
            return self.latest_status

    def _update_status(self, detections):
        status = "Face: None"
        for match, _face_location, _face_encoding in detections:
            if match:
                status = f"Recognized: {match}"
            else:
                status = "New Face Detected: Enter a name and register it."
                break

        with self.status_lock:
            self.latest_status = status

    def get_jpeg_frame(self):
        frame = self.get_frame()
        if frame is None:
            return None

        draw_detections(frame, self.get_detections())
        success, encoded_image = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not success:
            return None
        return encoded_image.tobytes()

    def release(self):
        self.stopped.set()
        self.capture_thread.join(timeout=1)
        self.recognition_thread.join(timeout=1)
        self.capture.release()


def register_new_face(name, face_encoding):
    """Save the latest unknown face encoding using the provided name."""
    registered_faces[name] = {
        "encoding": face_encoding,
    }
    refresh_known_face_cache()

    with open(REGISTERED_FACES_PATH, "wb") as f:
        pickle.dump(registered_faces, f)


def recognize_faces(frame):
    """Recognize faces on a smaller frame, then scale coordinates back up."""
    small_frame = cv2.resize(frame, (0, 0), fx=RECOGNITION_SCALE, fy=RECOGNITION_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    with known_faces_lock:
        names_snapshot = list(known_face_names)
        encodings_snapshot = list(known_face_encodings)

    recognized_faces = []
    scale = int(1 / RECOGNITION_SCALE)
    for face_encoding, face_location in zip(face_encodings, face_locations):
        match = None
        if encodings_snapshot:
            face_distances = face_recognition.face_distance(encodings_snapshot, face_encoding)
            best_match_index = int(np.argmin(face_distances))
            if face_distances[best_match_index] <= MATCH_TOLERANCE:
                match = names_snapshot[best_match_index]

        top, right, bottom, left = [coordinate * scale for coordinate in face_location]
        recognized_faces.append((match, (top, right, bottom, left), face_encoding))

    return recognized_faces


def draw_detections(frame, detections):
    """Draw the latest recognition result on the current video frame."""
    global temp_face_encoding

    for match, face_location, face_encoding in detections:
        top, right, bottom, left = face_location
        if match:
            color = (0, 255, 0)
            text = match
        else:
            color = (0, 0, 255)
            text = "New Face"
            with temp_face_lock:
                temp_face_encoding = face_encoding

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def generate_video_stream():
    """Yield multipart JPEG frames for the browser video feed."""
    frame_delay = 1 / STREAM_FPS
    while True:
        started_at = time.perf_counter()
        frame = video_processor.get_jpeg_frame()
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

        elapsed = time.perf_counter() - started_at
        time.sleep(max(0.001, frame_delay - elapsed))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    return jsonify({"status": video_processor.get_status()})


@app.route("/register", methods=["POST"])
def register():
    global temp_face_encoding

    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Please enter a name."}), 400

    with temp_face_lock:
        face_encoding = temp_face_encoding
        temp_face_encoding = None

    if face_encoding is None:
        return jsonify({"ok": False, "message": "No new face detected to register."}), 400

    register_new_face(name, face_encoding)
    return jsonify({"ok": True, "message": f"Registered {name}."})


video_processor = VideoProcessor().start()


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        video_processor.release()
