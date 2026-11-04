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
