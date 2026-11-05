import time
import cv2  # For image processing

from core.config import DEFAULT_SAMPLES_PER_USER, SAMPLE_INTERVAL_MS, ensure_directories
from core.database import DatabaseManager
from core.face_engine import FaceRecognitionEngine


# Create or update a user record in the SQLite database
def insertOrUpdate(user_id: int, name: str, age: int, department: str = "General"):
    db = DatabaseManager()
    return db.upsert_person(user_id, name, age, department=department)


def main():
    # Ensure dataSet/, dataSet/user/, and recognizer/ directories exist BEFORE connecting to DB
    ensure_directories()

    print("=== Enterprise Face Enrollment (CLI) ===")
    while True:
        try:
            user_id = int(input("Enter user ID (numeric): ").strip())
            if user_id <= 0:
                raise ValueError
            break
        except ValueError:
            print("Invalid ID. Please enter a positive integer ID.")

    while True:
        name = input("Enter user name: ").strip()
        if name:
            break
        print("Name cannot be empty.")

    while True:
        try:
            age = int(input("Enter user age: ").strip())
            if 1 <= age <= 130:
                break
            raise ValueError
        except ValueError:
            print("Invalid age. Please enter a numeric integer between 1 and 130.")

    department = input("Enter department (press Enter for 'General'): ").strip() or "General"

    # Insert or update the user information in the database
    insertOrUpdate(user_id, name, age, department)

    engine = FaceRecognitionEngine()

    cam = cv2.VideoCapture(0)  # Open the default camera
    if not cam.isOpened():
        raise RuntimeError("Error: Could not open the camera.")

    sample_num = 0  # Initialize the sample number for capturing images
    last_capture_ms = 0.0

    print(f"Capturing {DEFAULT_SAMPLES_PER_USER} face samples for {name} (Press 'q' to stop early)...")

    while True:
        ret, img = cam.read()  # Read a frame from the camera
        if not ret or img is None:
            print("Error: Failed to capture image from camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

        # Detect all faces in the frame (sorted largest-first so primary face is index 0)
        faces = engine.detect_faces(gray)

        now_ms = time.perf_counter() * 1000.0
        for idx, (x, y, w, h) in enumerate(faces):
            if idx == 0:
                # Capture only the primary (closest/largest) face to avoid bystander contamination
                if now_ms - last_capture_ms >= SAMPLE_INTERVAL_MS and sample_num < DEFAULT_SAMPLES_PER_USER:
                    sample_num += 1
                    engine.save_enrollment_sample(user_id, sample_num, gray, (x, y, w, h))
                    last_capture_ms = now_ms

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 160, 0), 2)
                cv2.putText(
                    img,
                    f"Capturing {name}: {sample_num}/{DEFAULT_SAMPLES_PER_USER}",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 50),
                    2,
                )
            else:
                # Highlight secondary faces in gray as ignored during single-user enrollment
                cv2.rectangle(img, (x, y), (x + w, y + h), (140, 140, 140), 1)
                cv2.putText(
                    img,
                    "Ignored (Bystander)",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                )

        cv2.imshow("Face Enrollment", img)  # Display the captured image with detected faces

        # Stop capturing after collecting enough samples or if the user presses 'q'
        if (cv2.waitKey(15) & 0xFF == ord("q")) or sample_num >= DEFAULT_SAMPLES_PER_USER:
            break

    cam.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print(f"Successfully saved {sample_num} normalized samples for {name} (ID: {user_id}).")


if __name__ == "__main__":
    main()
