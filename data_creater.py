import sqlite3  # For database operations

import cv2  # For image processing

# Load the built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)  # Open the default camera


# Create a database to store the face data
def insertOrUpdate(id, name, age):
    conn = sqlite3.connect("FaceBase.db")

    cmd = "SELECT * FROM People WHERE ID =" + str(id)
    cursor = conn.execute(cmd)  # Execute the SQL command to check if the record exists

    isRecordExist = 0  # Assume the record does not exist
    for row in cursor:
        isRecordExist = 1

    if isRecordExist == 1:
        cmd = "UPDATE People SET Name = ?, Age = ? WHERE ID = ?"
        conn.execute(cmd, (name, age, id))
    else:
        cmd = "INSERT INTO People(ID, Name, Age) VALUES(?, ?, ?)"
        conn.execute(cmd, (id, name, age))

    conn.commit()  # Close the database connection after committing the changes
    conn.close()  # Save all the changes


# User input for ID, Name, and Age
id = input("Enter user ID: ")
name = input("Enter user name: ")
age = input("Enter user age: ")

insertOrUpdate(id, name, age)  # Insert or update the user information in the database

sampleNum = 0  # Initialize the sample number for capturing images
while True:
    ret, img = cam.read()  # Read a frame from the camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    # ML model to detect faces in the image, Detect and scale factor of 1.3 and minimum neighbors of 5
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        sampleNum += 1  # Increment the sample number
        cv2.imwrite(
            "dataSet/User." + str(id) + "." + str(sampleNum) + ".jpg",
            gray[y : y + h, x : x + w],
        )  # Save the captured face image
        cv2.rectangle(
            img, (x, y), (x + w, y + h), (255, 0, 0), 2
        )  # Draw a rectangle around the detected face
        cv2.waitKey(100)  # Wait for a short period before capturing the next image

    cv2.imshow("Face", img)  # Display the captured image with detected faces
    cv2.waitKey(1)  # Wait for a key press

    if sampleNum > 20:  # Stop capturing after collecting enough samples
        break
