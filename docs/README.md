# FRS Project Documentation

Welcome to the documentation for **FRS**, a Flask-based face recognition system that captures webcam video, detects faces, recognizes registered people, and lets users register newly detected faces from a browser interface.

## Documentation Index

- [Project Overview](./project-overview.md) - Purpose, features, repository structure, and high-level behavior.
- [Architecture](./architecture.md) - Component layout, threading model, data flow, and runtime lifecycle.
- [Backend Reference](./backend-reference.md) - Flask routes, recognition pipeline, constants, globals, and persistence behavior.
- [Frontend Reference](./frontend-reference.md) - HTML, CSS, JavaScript, UI behavior, and browser-to-server interactions.
- [Setup and Running](./setup-and-running.md) - Requirements, installation, platform notes, and run instructions.
- [Data Storage and Privacy](./data-storage-and-privacy.md) - Face encoding storage, generated files, privacy considerations, and safe handling.
- [Operations and Troubleshooting](./operations-and-troubleshooting.md) - Common runtime issues, camera problems, dependency problems, and tuning guidance.
- [Development Guide](./development-guide.md) - Code organization, contribution workflow, testing ideas, and extension points.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

After the server starts, open `http://localhost:5000` in a browser with access to the machine running the webcam.

> Note: This project requires a working camera device and native dependencies used by OpenCV, dlib, and `face_recognition`.
