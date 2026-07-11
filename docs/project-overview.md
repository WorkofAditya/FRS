# Project Overview

## What FRS Does

FRS is a real-time face recognition web application. It uses a local webcam to capture video, analyzes frames for faces, compares detected face encodings against previously registered encodings, and streams annotated video to a browser.

The application can:

- Capture live webcam frames.
- Detect faces in real time.
- Compare unknown faces with registered faces.
- Draw visual labels around recognized and unknown faces.
- Expose a browser UI for viewing the camera stream.
- Register a newly detected face by submitting a name.
- Persist registered face encodings to a local pickle file.

## Repository Structure

```text
FRS/
├── app.py                  # Flask server, webcam processing, recognition, registration, and streaming
├── requirements.txt        # Python dependencies
├── README.md               # Public project summary and installation notes
├── LICENSE                 # MIT license
├── static/
│   ├── script.js           # Browser-side status polling and registration form handling
│   └── style.css           # Browser UI styling
├── templates/
│   └── index.html          # Main Flask-rendered page
└── docs/                   # Detailed project documentation
```

## Main Technologies

- **Python** powers the backend application.
- **Flask** serves the web page, status API, registration API, and MJPEG video stream.
- **OpenCV** captures webcam frames, resizes images, converts color spaces, draws labels, and JPEG-encodes frames.
- **face_recognition** detects faces and produces numerical face encodings.
- **NumPy** helps select the closest face match using vector distances.
- **Pickle** stores registered face encodings in `registered_faces.pkl`.
- **HTML/CSS/JavaScript** provide the browser interface.

## User Workflow

1. Start the Flask application with `python app.py`.
2. The backend opens the default camera and starts background capture and recognition threads.
3. The user opens the web page in a browser.
4. The browser displays the `/video` MJPEG stream.
5. The browser polls `/status` every 500 milliseconds.
6. If a known face appears, the video overlay and status show the registered name.
7. If an unknown face appears, the video overlay shows `New Face` and the status asks for registration.
8. The user enters a name and submits the registration form.
9. The backend stores the latest unknown face encoding under that name.

## Current Scope

The project is designed as a local, single-process face recognition demo or lightweight application. It does not include multi-user accounts, authentication, database migrations, cloud storage, or production deployment configuration.
