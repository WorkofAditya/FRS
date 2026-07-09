import os
import pickle
import queue
import threading
import tkinter as tk
from tkinter import simpledialog

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk

REGISTERED_FACES_PATH = "registered_faces.pkl"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RECOGNITION_SCALE = 0.25
RECOGNITION_INTERVAL = 3
FRAME_QUEUE_SIZE = 1

# Load registered faces from file
if os.path.exists(REGISTERED_FACES_PATH):
    with open(REGISTERED_FACES_PATH, "rb") as f:
        registered_faces = pickle.load(f)
else:
    registered_faces = {}

# Global variable to temporarily hold face encoding for new faces
temp_face_encoding = None


def known_face_data():
    """Return cached names and encodings for faster vectorized matching."""
    names = list(registered_faces.keys())
    encodings = [info["encoding"] for info in registered_faces.values()]
    return names, encodings


known_face_names, known_face_encodings = known_face_data()


class WebcamStream:
    """Continuously read webcam frames and keep only the newest frame.

    OpenCV can buffer frames faster than face recognition can process them. Keeping
    a single latest-frame queue prevents the UI from showing old frames several
    seconds late when recognition work is temporarily slow.
    """

    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frames = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped.is_set():
            ret, frame = self.capture.read()
            if not ret:
                continue

            if self.frames.full():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
            self.frames.put(frame)

    def read(self):
        try:
            return self.frames.get_nowait()
        except queue.Empty:
            return None

    def release(self):
        self.stopped.set()
        self.thread.join(timeout=1)
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
    global known_face_names, known_face_encodings

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
    known_face_names, known_face_encodings = known_face_data()

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

    recognized_faces = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        match = None
        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = int(np.argmin(face_distances))
            if face_distances[best_match_index] <= 0.5:
                match = known_face_names[best_match_index]

        scale = int(1 / RECOGNITION_SCALE)
        top, right, bottom, left = [coordinate * scale for coordinate in face_location]
        recognized_faces.append((match, (top, right, bottom, left), face_encoding))

    return recognized_faces


def main():
    global temp_face_encoding

    stream = WebcamStream().start()

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

    frame_count = 0
    last_faces = []

    def update_frame():
        nonlocal frame_count, last_faces
        global temp_face_encoding

        frame = stream.read()
        if frame is not None:
            frame_count += 1

            # Face encoding is the expensive part. Do it on a small frame and only
            # every few camera frames; draw cached boxes between recognition runs.
            if frame_count % RECOGNITION_INTERVAL == 0:
                last_faces = recognize_faces(frame)

            face_label_text = "Face: None"
            for match, face_location, face_encoding in last_faces:
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

            face_label.config(text=face_label_text)

            # Convert frame to image and update the GUI
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)

            video_label.config(image=imgtk)
            video_label.image = imgtk

        video_label.after(1, update_frame)

    def on_close():
        stream.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    update_frame()
    root.mainloop()


if __name__ == "__main__":
    main()
