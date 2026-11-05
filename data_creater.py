import os  # For directory and file path operations
import sqlite3  # For database operations

import cv2  # For image processing

# Load the built-in face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():  # Check if the cascade classifier loaded successfully
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


# Create a database to store the face data
def insertOrUpdate(id, name, age):
    conn = sqlite3.connect("dataset/FaceBase.db")

    # Ensure the People table exists before querying
    conn.execute(
        "CREATE TABLE IF NOT EXISTS People (ID INTEGER PRIMARY KEY, Name VARCHAR(20), Age INTEGER)"
    )

    cmd = "SELECT * FROM People WHERE ID = ?"
    cursor = conn.execute(
        cmd, (id,)
    )  # Execute the SQL command to check if the record exists

    isRecordExist = 0  # Assume the record does not exist
    for row in cursor:
        isRecordExist = 1

    if isRecordExist == 1:
        cmd = "UPDATE People SET Name = ?, Age = ? WHERE ID = ?"
        conn.execute(cmd, (name, age, id))
    else:
        cmd = "INSERT INTO People(ID, Name, Age) VALUES(?, ?, ?)"
        conn.execute(cmd, (id, name, age))

    conn.commit()  # Save all the changes
    conn.close()  # Close the database connection after committing the changes


# ____Data Collection____

# User input for ID, Name, and Age
while True:
    try:
        id = int(input("Enter user ID: "))
        break
    except ValueError:
        print("Invalid ID. Please enter a numeric integer ID.")

name = input("Enter user name: ").strip()

while True:
    try:
        age = int(input("Enter user age: "))
        break
    except ValueError:
        print("Invalid age. Please enter a numeric integer age.")

insertOrUpdate(id, name, age)  # Insert or update the user information in the database

os.makedirs("dataSet", exist_ok=True)  # Ensure the dataSet directory exists

cam = cv2.VideoCapture(0)  # Open the default camera
if not cam.isOpened():
    raise RuntimeError("Error: Could not open the camera.")

sampleNum = 0  # Initialize the sample number for capturing images
while True:
    ret, img = cam.read()  # Read a frame from the camera
    if not ret or img is None:
        print("Error: Failed to capture image from camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    # ML model to detect faces in the image, Detect and scale factor of 1.3 and minimum neighbors of 5
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        sampleNum += 1  # Increment the sample number
        cv2.imwrite(
            "dataSet/user/User." + str(id) + "." + str(sampleNum) + ".jpg",
            gray[y : y + h, x : x + w],
        )  # Save the captured face image
        cv2.rectangle(
            img, (x, y), (x + w, y + h), (255, 0, 0), 2
        )  # Draw a rectangle around the detected face
        cv2.waitKey(500)  # Wait for a short period before capturing the next image

        if sampleNum >= 50:
            break

    cv2.imshow("Face", img)  # Display the captured image with detected faces

    # Stop capturing after collecting 50 samples or if the user presses 'q'
    if (cv2.waitKey(1) & 0xFF == ord("q")) or sampleNum >= 50:
        break

cam.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows
