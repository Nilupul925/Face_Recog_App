import sqlite3

import cv2

face_detect = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)  # Load the built-in face detector

cam = cv2.VideoCapture(0)  # Start the webcam for capturing video

recognizer = (
    cv2.face.LBPHFaceRecognizer_create()
)  # Create the Local Binary Patterns Histograms (LBPH) face recognizer
recognizer.read("recognizer/training_data.yml")  # Load the trained model from a file


def detect_face(id):
    conn = sqlite3.connect("dataset/FaceBase.db")  # Connect to the database
    cursor = conn.execute(
        "SELECT * FROM People WHERE ID = ?", (id,)
    )  # Execute a query to fetch records from the People table

    profile = None  # Initialize the profile variable to None
    for row in cursor:
        profile = row  # If a record is found, assign it to the profile variable
    conn.close()  # Close the database connection

    return profile  # Return the profile (record) if found, otherwise return None


while True:
    ret, img = cam.read()  # Read a frame from the webcam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    faces = face_detect.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )  # Detect faces in the grayscale frame

    for x, y, w, h in faces:
        cv2.rectangle(
            img, (x, y), (x + w, y + h), (0, 255, 0), 2
        )  # Draw a rectangle around the detected face

        id, confidence = recognizer.predict(
            gray[y : y + h, x : x + w]
        )  # Predict the ID of the detected face

        profile = detect_face(
            id
        )  # Fetch the profile (record) from the database based on the predicted ID

        if confidence < 100:  # If confidence is less than 100, consider it a match
            name = f"Name: {profile[1]}"  # Format the name for display
            age = f"Age: {profile[2]}"  # Format the age for display
            matching = f"{round(100 - confidence)}%"  # Calculate and format the confidence percentage
        else:
            name = "Unknown"  # If not a match, label as Unknown
            age = "Age: Unknown"  # If not a match, label age as Unknown
            matching = f"{round(100 - confidence)}%"  # Calculate and format the confidence percentage

        cv2.putText(
            img,
            str(name),
            (x + 5, y + h + 25),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            1,
        )  # Display the name on the frame
        cv2.putText(
            img,
            str(age),
            (x + 5, y + h + 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            1,
        )  # Display the age on the frame
        # cv2.putText(
        #     img,
        #     str(matching),
        #     (x + 5, y + h + 75),
        #     cv2.FONT_HERSHEY_COMPLEX,
        #     1,
        #     (255, 255, 255),
        #     1,
        # )  # Display the confidence percentage on the frame

        cv2.rectangle(
            img, (x, y), (x + w, y + h), (0, 255, 0), 2
        )  # Draw a rectangle around the detected face

    cv2.imshow("Face", img)  # Display the frame with detected faces

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit if 'q' is pressed
        break

cam.release()  # Release the webcam resource
cv2.destroyAllWindows()  # Close all OpenCV windows
