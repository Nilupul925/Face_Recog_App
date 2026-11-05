import time
from typing import Optional

import cv2

from core.database import DatabaseManager, PersonProfile
from core.face_engine import FaceRecognitionEngine, draw_enterprise_hud

# Initialize shared enterprise database & recognition engine (with in-memory profile caching)
db_manager = DatabaseManager()
engine = FaceRecognitionEngine(db_manager)


def detect_face(person_id: int) -> Optional[PersonProfile]:
    """Fast O(1) in-memory profile lookup by person ID."""
    return db_manager.get_person(person_id)


def main():
    if not engine.model_loaded:
        print(
            "Warning: Trained model 'recognizer/training_data.yml' not found. "
            "Faces will be detected as Unknown until you run 'python trainer.py' or train via 'python app.py'."
        )

    cam = cv2.VideoCapture(0)  # Start the webcam for capturing video
    if not cam.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    fps = 0.0
    last_ts = time.perf_counter()

    print("Simultaneous Multi-Person Face Recognition running... Press 'q' to exit.")

    while True:
        ret, img = cam.read()  # Read a frame from the webcam
        if not ret or img is None:
            print("Warning: Failed to read frame from webcam.")
            break

        now = time.perf_counter()
        dt = max(now - last_ts, 1e-4)
        last_ts = now
        fps = (0.85 * fps + 0.15 * (1.0 / dt)) if fps > 0 else (1.0 / dt)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

        # Detect and recognize ALL faces simultaneously in the current frame
        detections = engine.recognize_faces_in_frame(gray, record_attendance=True)

        # Render modern HUD boxes, names, ages, departments, and match percentages
        annotated = draw_enterprise_hud(
            img,
            detections,
            fps=fps,
            model_loaded=engine.model_loaded,
        )

        cv2.imshow("VisionGuard — Simultaneous Multi-Person Face Recognition", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit if 'q' is pressed
            break

    cam.release()  # Release the webcam resource
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    main()
