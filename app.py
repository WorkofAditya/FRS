from flask import Flask, render_template, Response, request
import cv2
import face_recognition
import pickle
import os
import numpy as np
import eventlet
import eventlet.wsgi

app = Flask(__name__)

# Load registered faces
if os.path.exists("registered_faces.pkl"):
    with open("registered_faces.pkl", "rb") as f:
        registered_faces = pickle.load(f)
else:
    registered_faces = {}

known_face_encodings = [v["encoding"] for v in registered_faces.values()]
known_face_names = list(registered_faces.keys())

temp_face_encoding = None
video_source = 0
capture = cv2.VideoCapture(video_source)

process_this_frame = True

def generate_frames():
    global temp_face_encoding, process_this_frame

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        face_encodings = []

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        process_this_frame = not process_this_frame

        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "New Face"

            if known_face_encodings:
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.45:
                    name = known_face_names[best_match_index]
                else:
                    temp_face_encoding = face_encoding

            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0,255,0) if name != "New Face" else (0,0,255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    global temp_face_encoding, known_face_encodings, known_face_names

    name = request.form.get("name")
    if name and temp_face_encoding is not None:
        registered_faces[name] = {"encoding": temp_face_encoding}

        known_face_encodings.append(temp_face_encoding)
        known_face_names.append(name)

        with open("registered_faces.pkl", "wb") as f:
            pickle.dump(registered_faces, f)

        temp_face_encoding = None
        return "Face registered"

    return "No face to register", 400

if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 5000)), app)
