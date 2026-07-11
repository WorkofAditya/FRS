# Frontend Reference

## Overview

The frontend is intentionally lightweight. Flask renders a single HTML page, static CSS styles the interface, and a small JavaScript file handles status polling and registration form submission.

## HTML Template

`templates/index.html` defines the page structure:

- Page title and stylesheet link.
- Main application shell.
- Status text area.
- MJPEG video feed displayed through an `<img>` element.
- Registration form with a name input.
- Message area for success and error feedback.
- Script include for frontend behavior.

The video feed uses:

```html
<img src="/video" id="video-feed" alt="Live face recognition video feed">
```

Because `/video` returns multipart MJPEG data, the browser keeps updating the image as new frames arrive.

## JavaScript Behavior

`static/script.js` performs three tasks:

1. Handles the registration form submit event.
2. Sends the entered name to `/register` as URL-encoded form data.
3. Polls `/status` every 500 milliseconds and updates the status text.

## Registration Flow

1. User enters a name.
2. User clicks **Register Face**.
3. JavaScript prevents normal form navigation.
4. Empty names are rejected in the browser.
5. A `POST /register` request is sent with `Content-Type: application/x-www-form-urlencoded`.
6. The JSON response controls the feedback message.
7. On success, the input field is cleared.

## Status Polling

`refreshStatus()` fetches `/status` with `cache: 'no-store'` so the browser does not reuse stale responses. If the request fails, the UI shows `Waiting for camera stream...`.

## Styling

`static/style.css` creates a centered dark interface with:

- White text on a dark background.
- Green accent color for active status and buttons.
- A glowing green border around the video feed.
- Responsive wrapping for the registration form.
- Separate success and error message colors.

## Browser Requirements

The frontend should work in modern browsers that support:

- `fetch()`.
- `URLSearchParams`.
- Multipart MJPEG display through an `<img>` element.

The browser does not directly access the webcam. Camera access happens on the server machine through OpenCV.
