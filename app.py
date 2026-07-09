import os
import pickle
import threading
import time
import tkinter as tk
from tkinter import simpledialog

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk

REGISTERED_FACES_PATH = "registered_faces.pkl"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISPLAY_FPS = 30
RECOGNITION_SCALE = 0.25
RECOGNITION_FPS = 5
MATCH_TOLERANCE = 0.5

# Load registered faces from file
if os.path.exists(REGISTERED_FACES_PATH):
    with open(REGISTERED_FACES_PATH, "rb") as f:
        registered_faces = pickle.load(f)
else:
    registered_faces = {}

# Global variable to temporarily hold face encoding for new faces
temp_face_encoding = None
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
    """Capture webcam frames quickly while recognizing faces in the background.

    The display loop never performs face encoding. One thread only reads the newest
    webcam frame, and another slower thread analyzes occasional snapshots. This
    keeps the video smooth even if face recognition takes longer than one frame.
    """

    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_lock = threading.Lock()
        self.detections_lock = threading.Lock()
        self.stopped = threading.Event()
        self.latest_frame = None
        self.latest_detections = []

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)

    def start(self):
        self.capture_thread.start()
        self.recognition_thread.start()
        return self

    def _capture_loop(self):
        while not self.stopped.is_set():
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.005)
                continue

            with self.frame_lock:
                self.latest_frame = frame

    def _recognition_loop(self):
        recognition_delay = 1 / RECOGNITION_FPS
        while not self.stopped.is_set():
            started_at = time.perf_counter()
            frame = self.get_frame()
            if frame is not None:
                detections = recognize_faces(frame)
                with self.detections_lock:
                    self.latest_detections = detections

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

    def release(self):
        self.stopped.set()
        self.capture_thread.join(timeout=1)
        self.recognition_thread.join(timeout=1)
        self.capture.release()


def capture_new_face():
    """Called when the user clicks the 'Capture Face' button to store a new face."""
    global temp_face_encoding
    if temp_face_encoding is not None:
        register_new_face(temp_face_encoding)
        temp_face_encoding = None  # Clear temporary storage after registering the face
    else:
        print("No new face detected to capture!")


def register_new_face(face_encoding):
    root = tk.Toplevel()
    root.withdraw()

    # Ask for user name only to associate with the new face
    name = simpledialog.askstring("New Face Detected", "Enter your name:", parent=root)
    if name is None:
        root.destroy()
        return

    # Store the new face data with just the name
    registered_faces[name] = {
        "encoding": face_encoding,
    }
    refresh_known_face_cache()

    # Save the updated registered faces to the file
    with open(REGISTERED_FACES_PATH, "wb") as f:
        pickle.dump(registered_faces, f)

    root.destroy()


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

    face_label_text = "Face: None"
    for match, face_location, face_encoding in detections:
        top, right, bottom, left = face_location
        if match:
            color = (0, 255, 0)
            text = match
            face_label_text = f"Recognized: {match}"
        else:
            color = (0, 0, 255)
            text = "New Face"
            face_label_text = "New Face Detected: Capture it?"
            temp_face_encoding = face_encoding

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return face_label_text


def main():
    processor = VideoProcessor().start()

    root = tk.Tk()
    root.title("Face Recognition System")

    # Video feed label
    video_label = tk.Label(root)
    video_label.pack()

    # Label to display recognized face details
    face_label = tk.Label(root, text="Face: None", font=("Helvetica", 14))
    face_label.pack()

    # Button to capture and save new face
    capture_button = tk.Button(root, text="Capture Face", command=capture_new_face, font=("Helvetica", 14))
    capture_button.pack(pady=10)

    frame_delay_ms = max(1, int(1000 / DISPLAY_FPS))

    def update_frame():
        frame = processor.get_frame()
        if frame is not None:
            face_label_text = draw_detections(frame, processor.get_detections())
            face_label.config(text=face_label_text)

            # Convert frame to image and update the GUI
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)

            video_label.config(image=imgtk)
            video_label.image = imgtk

        video_label.after(frame_delay_ms, update_frame)

    def on_close():
        processor.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    update_frame()
    root.mainloop()


if __name__ == "__main__":
    main()
