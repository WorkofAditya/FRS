# Operations and Troubleshooting

## Common Startup Problems

### `ModuleNotFoundError`

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure you are using the same Python environment that runs `app.py`.

### `face_recognition` or `dlib` Build Failure

`face_recognition` depends on native packages. Install platform build tools:

- Linux: compiler, Python development headers, CMake, and related native libraries.
- Windows: Visual Studio Build Tools with Desktop development with C++.

### Camera Does Not Open

The app uses `cv2.VideoCapture(0)`, which selects the default camera. Check that:

- A camera is connected.
- Another application is not already using the camera.
- The process has camera permissions.
- The correct camera index is used if multiple cameras exist.

To use another camera, instantiate `VideoProcessor` with a different source index or video source.

## Runtime Problems

### Video Stream Is Blank

Possible causes:

- Camera frames are not being captured.
- Browser cannot reach the Flask server.
- OpenCV cannot access the camera device.
- The server process is running in an environment without camera passthrough.

### Recognition Is Slow

Tune these constants in `app.py`:

- Lower `RECOGNITION_FPS` to analyze fewer frames per second.
- Lower `RECOGNITION_SCALE` for faster recognition with less detail.
- Lower `FRAME_WIDTH` and `FRAME_HEIGHT` to reduce captured frame size.

### False Matches

Lower `MATCH_TOLERANCE` to make matching stricter. This can reduce false positives but may increase false negatives.

### Known Person Not Recognized

Possible fixes:

- Register the person again with better lighting.
- Face the camera directly during registration.
- Improve lighting and reduce motion blur.
- Raise `MATCH_TOLERANCE` slightly if matching is too strict.

### Registration Says No New Face Detected

The backend only registers the latest unknown face encoding. Make sure:

- A face is visible in the stream.
- The status says a new face was detected.
- The person is not already matched as a registered face.
- You submit the form soon after the unknown face appears.

## Performance Tuning

| Goal | Suggested Change | Tradeoff |
| --- | --- | --- |
| Faster recognition | Decrease `RECOGNITION_FPS` | Less frequent status updates |
| Lower CPU usage | Decrease capture or stream FPS | Less smooth video |
| Better visual quality | Increase `JPEG_QUALITY` | Larger stream bandwidth |
| Stricter matching | Lower `MATCH_TOLERANCE` | More missed matches |
| More permissive matching | Raise `MATCH_TOLERANCE` | More false matches |

## Production Readiness Checklist

Before production use, consider adding:

- Authentication and authorization.
- HTTPS termination.
- CSRF protection for registration.
- Safer storage than pickle.
- A way to delete registered users.
- Structured logging.
- Health checks.
- Configuration through environment variables.
- Automated tests.
- Clear privacy and retention policies.
