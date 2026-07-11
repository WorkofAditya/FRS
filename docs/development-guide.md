# Development Guide

## Code Organization

The project is intentionally compact:

- `app.py` contains backend configuration, face recognition logic, streaming, registration, and Flask routes.
- `templates/index.html` contains the single rendered page.
- `static/script.js` contains browser behavior.
- `static/style.css` contains UI styling.
- `requirements.txt` lists Python dependencies.
- `docs/` contains this documentation set.

## Local Development Workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

After making changes, manually verify:

1. The server starts without import errors.
2. The browser loads `/`.
3. The video feed renders.
4. `/status` returns JSON.
5. Registration succeeds when an unknown face is visible.
6. `registered_faces.pkl` is created or updated.

## Suggested Automated Checks

This repository currently does not include a dedicated test suite. Useful future checks include:

- Python syntax compilation with `python -m py_compile app.py`.
- Unit tests for registration and recognition helper functions with mocked encodings.
- Flask route tests using Flask's test client.
- Frontend smoke checks for expected DOM elements.
- Formatting and linting with tools such as Ruff or Black.

## Extension Points

### Configuration

Move constants from `app.py` into environment variables or a config file to make deployment easier.

### Storage

Replace `registered_faces.pkl` with SQLite or another database if you need safer querying, deletion, migrations, or metadata.

### Identity Management

Add routes and UI for:

- Listing registered people.
- Deleting a registered person.
- Updating names.
- Exporting or backing up registrations.

### Security

Add authentication before exposing the app beyond a trusted local environment. Registration currently accepts any form submission that can reach the server.

### Recognition Models

The current face location model is `hog`, which is CPU-friendly. Systems with GPU support could experiment with CNN-based detection, but this requires additional native setup and more compute.

### API Design

If the frontend grows, consider documenting and versioning the JSON endpoints, for example under `/api/status` and `/api/register`.

## Contribution Guidelines

When contributing:

- Keep changes focused and easy to review.
- Update documentation when behavior changes.
- Avoid committing generated biometric data such as `registered_faces.pkl`.
- Test with an actual camera when changing recognition or streaming behavior.
- Be careful with threading changes and protect shared mutable state with locks.
