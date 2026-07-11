# Setup and Running

## Requirements

- Python 3.11 is recommended by the project README.
- A working webcam connected to the machine running the Flask app.
- Native build tools required by `face_recognition` and its dependencies.
- Python packages listed in `requirements.txt`:
  - `cmake`
  - `opencv-python`
  - `face-recognition`
  - `numpy`
  - `Pillow`
  - `flask`

## Recommended Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open the application at:

```text
http://localhost:5000
```

If you run the server on another machine in the same network, open `http://<server-ip>:5000`.

## Linux Notes

Some Linux systems need OpenCV runtime libraries. If camera display or import fails with missing OpenGL-related libraries, install the appropriate system package for your distribution. On Debian or Ubuntu-based systems, this is commonly:

```bash
sudo apt install libgl1-mesa-glx
```

Some newer distributions provide the package as `libgl1` instead.

## Windows Notes

The `face_recognition` package depends on native components. Windows users usually need:

- Python 3.11 64-bit.
- Visual Studio Build Tools.
- The **Desktop development with C++** workload.
- A camera available to desktop applications.

## Running the App

Use:

```bash
python app.py
```

The app starts Flask on `0.0.0.0:5000`, which means it listens on all network interfaces. For local use, browse to `http://localhost:5000`.

## Generated Files

The app may generate:

```text
registered_faces.pkl
```

This file stores registered face encodings and should be treated as sensitive biometric data.

## Stopping the App

Press `Ctrl+C` in the terminal. The application attempts to release the camera in the `finally` block when running as the main script.
