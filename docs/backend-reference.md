# Backend Reference

## Entry Point

The backend lives in `app.py`. Running `python app.py` starts the Flask development server on `0.0.0.0:5000` and initializes webcam processing.

## Configuration Constants

| Constant | Purpose |
| --- | --- |
| `REGISTERED_FACES_PATH` | File path used to persist registered face encodings. |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Requested webcam capture resolution. |
| `CAPTURE_FPS` | Target camera capture rate. |
| `STREAM_FPS` | Target MJPEG streaming rate. |
| `JPEG_QUALITY` | JPEG encoding quality for streamed frames. |
| `RECOGNITION_SCALE` | Downscale factor used before recognition for speed. |
| `RECOGNITION_FPS` | Target recognition loop rate. |
| `MATCH_TOLERANCE` | Maximum face distance accepted as a match. Lower values are stricter. |

## Persistent Registry

At startup, the application looks for `registered_faces.pkl` in the working directory. If it exists, the file is loaded with `pickle`. If it does not exist, the app starts with an empty registry.

The registry structure is:

```python
{
    "Person Name": {
        "encoding": <face encoding vector>
    }
}
```

When a new face is registered, the registry is written back to `registered_faces.pkl`.

## Known-Face Cache

`refresh_known_face_cache()` converts the registry dictionary into two lists:

- `known_face_names`
- `known_face_encodings`

These lists make recognition matching simpler and faster because the code can compare a detected encoding against all cached known encodings.

## `VideoProcessor`

`VideoProcessor` owns the webcam capture object and the background processing threads.

### Responsibilities

- Open and configure the camera.
- Capture frames continuously.
- Run recognition periodically.
- Store the latest frame, detections, and status.
- Produce annotated JPEG frames for the video stream.
- Release the camera on shutdown.

### Important Methods

| Method | Description |
| --- | --- |
| `start()` | Starts capture and recognition daemon threads. |
| `_capture_loop()` | Reads camera frames and stores the latest frame. |
| `_recognition_loop()` | Runs face recognition against the latest frame. |
| `get_frame()` | Returns a copy of the latest frame. |
| `get_detections()` | Returns the most recent detection list. |
| `get_status()` | Returns the current status text. |
| `_update_status()` | Converts detections into user-facing status text. |
| `get_jpeg_frame()` | Draws detections and returns a JPEG-encoded frame. |
| `release()` | Stops processing and releases camera resources. |

## Face Registration

`register_new_face(name, face_encoding)` stores the supplied encoding in `registered_faces`, refreshes the known-face cache, and serializes the registry to disk.

The `/register` route uses `temp_face_encoding`, which is set when the rendering layer sees an unknown face in the latest detections. Registration can fail if:

- The submitted name is empty.
- No new face encoding is currently available.

## Recognition Pipeline

`recognize_faces(frame)` performs the following steps:

1. Resize the frame using `RECOGNITION_SCALE`.
2. Convert the resized frame from BGR to RGB.
3. Locate faces with `face_recognition.face_locations(..., model="hog")`.
4. Generate encodings with `face_recognition.face_encodings(...)`.
5. Compare each encoding against cached known encodings using face distance.
6. Select the closest known face with `numpy.argmin`.
7. Accept the match only when the distance is within `MATCH_TOLERANCE`.
8. Scale face coordinates back to the original frame size.
9. Return tuples of `(match, face_location, face_encoding)`.

## Detection Rendering

`draw_detections(frame, detections)` draws a rectangle and label for each detected face:

- Green rectangle and registered name for known faces.
- Red rectangle and `New Face` for unknown faces.

When an unknown face is rendered, its encoding is stored as `temp_face_encoding` so it can be registered by the form.

## Flask Routes

| Route | Method | Response | Purpose |
| --- | --- | --- | --- |
| `/` | `GET` | HTML | Renders the main web UI. |
| `/video` | `GET` | Multipart MJPEG stream | Streams annotated camera frames. |
| `/status` | `GET` | JSON | Returns current face recognition status. |
| `/register` | `POST` | JSON | Registers the latest unknown face with a submitted name. |

### `/status` Response Example

```json
{
  "status": "Recognized: Ada"
}
```

### `/register` Request Example

```http
POST /register
Content-Type: application/x-www-form-urlencoded

name=Ada
```

### `/register` Success Example

```json
{
  "ok": true,
  "message": "Registered Ada."
}
```
