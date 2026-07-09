import cv2
import face_recognition
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

# Load registered faces from file
if os.path.exists("registered_faces.pkl"):
    with open("registered_faces.pkl", "rb") as f:
        registered_faces = pickle.load(f)
else:
    registered_faces = {}

# Global variable to temporarily hold face encoding for new faces
temp_face_encoding = None

def capture_new_face():
    """Called when the user clicks the 'Capture Face' button to store a new face."""
    global temp_face_encoding
    if temp_face_encoding is not None:
        register_new_face(temp_face_encoding)
        temp_face_encoding = None  # Clear temporary storage after registering the face
    else:
        print("No new face detected to capture!")

def register_new_face(face_encoding):
    root = tk.Tk()
    root.withdraw()

    # Ask for user name only to associate with the new face
    name = simpledialog.askstring("New Face Detected", "Enter your name:")
    if name is None:
        root.destroy()
        return

    # Store the new face data with just the name
    registered_faces[name] = {
        "encoding": face_encoding,
    }

    # Save the updated registered faces to the file
    with open("registered_faces.pkl", "wb") as f:
        pickle.dump(registered_faces, f)

    root.destroy()

def main():
    global temp_face_encoding

    capture = cv2.VideoCapture(0)

    root = tk.Tk()
    root.title("Face Recognition System")

    # Video feed label
    video_label = tk.Label(root)
    video_label.pack()

    # Label to display recognized face details
    face_label = tk.Label(root, text="Face: None", font=('Helvetica', 14))
    face_label.pack()

    # Button to capture and save new face
    capture_button = tk.Button(root, text="Capture Face", command=capture_new_face, font=('Helvetica', 14))
    capture_button.pack(pady=10)

    def update_frame():
        global temp_face_encoding
        ret, frame = capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                match = None
                for name, info in registered_faces.items():
                    matches = face_recognition.compare_faces([info['encoding']], face_encoding, tolerance=0.5)
                    if matches[0]:
                        match = name
                        break

                top, right, bottom, left = face_location
                if match:
                    # Draw a box around the recognized face and display the name
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, match, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    face_label.config(text=f"Recognized: {match}")
                else:
                    # Draw a red box around an unrecognized face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "New Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    face_label.config(text="New Face Detected: Capture it?")
                    
                    # Temporarily store the new face encoding
                    temp_face_encoding = face_encoding

            # Convert frame to image and update the GUI
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)

            video_label.config(image=imgtk)
            video_label.image = imgtk

        video_label.after(10, update_frame)

    update_frame()

    root.mainloop()

    capture.release()

if __name__ == '__main__':
    main()
