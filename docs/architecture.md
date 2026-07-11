# Architecture

## High-Level Architecture

FRS is a single Flask application with three major layers:

1. **Camera and recognition layer** in `app.py`.
2. **HTTP API and streaming layer** in `app.py`.
3. **Browser interface layer** in `templates/index.html`, `static/script.js`, and `static/style.css`.

```text
Webcam
  │
  ▼
VideoProcessor capture thread
  │ stores latest frame
  ▼
VideoProcessor recognition thread
  │ analyzes periodic frames
  ▼
latest detections + status
  │
  ├── /video    -> annotated MJPEG stream
  ├── /status   -> JSON status text
  └── /register -> saves latest unknown face encoding
```

## Runtime Lifecycle

1. Python imports `app.py`.
2. The app checks for `registered_faces.pkl`.
3. Registered face encodings are loaded if the file exists; otherwise an empty registry is created.
4. The known-face cache is initialized.
5. `VideoProcessor().start()` opens the default camera and starts two daemon threads.
6. Flask serves routes when `app.py` is executed directly.
7. On application shutdown, `video_processor.release()` attempts to stop threads and release the camera.

## Threading Model

The application separates capture, recognition, and streaming so the live feed remains responsive even when face recognition is slower than camera capture.

### Capture Thread

The capture thread continuously reads frames from OpenCV's `VideoCapture` object and stores the latest frame in memory. It targets `CAPTURE_FPS` and uses a lock to protect frame access.

### Recognition Thread

The recognition thread periodically copies the latest frame, downsizes it, detects faces, computes encodings, compares them to known encodings, and updates the latest detection list and status text. It targets `RECOGNITION_FPS`, which is lower than the stream frame rate to reduce CPU usage.

### Request Threads

Flask handles HTTP requests. Route handlers read shared state through lock-protected methods and return HTML, JSON, or streaming frame data.

## Shared State

| State | Purpose | Protection |
| --- | --- | --- |
| `registered_faces` | Persistent in-memory mapping of names to encodings | Refreshed through helper functions |
| `known_face_names` | Cached list of registered names for matching | `known_faces_lock` |
| `known_face_encodings` | Cached list of registered encodings for matching | `known_faces_lock` |
| `temp_face_encoding` | Latest unknown face encoding available for registration | `temp_face_lock` |
| `latest_frame` | Most recent webcam frame | `frame_lock` |
| `latest_detections` | Most recent recognition results | `detections_lock` |
| `latest_status` | Text shown by `/status` | `status_lock` |

## Face Recognition Data Flow

1. A full-size BGR camera frame is captured by OpenCV.
2. Recognition resizes the frame by `RECOGNITION_SCALE`.
3. The resized frame is converted from BGR to RGB for `face_recognition`.
4. Face locations are found using the HOG model.
5. Face encodings are computed for detected locations.
6. Each encoding is compared with the cached registered encodings.
7. The closest match is accepted only if its distance is less than or equal to `MATCH_TOLERANCE`.
8. Small-frame coordinates are scaled back up to the displayed frame size.
9. Detections are cached for rendering and status updates.

## Video Streaming Flow

The `/video` route returns a multipart MJPEG response. Each loop iteration:

1. Copies the latest frame.
2. Draws the latest detections on top of the copied frame.
3. Encodes the frame as JPEG.
4. Yields the encoded bytes with the multipart boundary expected by browsers.

This lets a normal `<img>` element display a continuously updating camera stream.
