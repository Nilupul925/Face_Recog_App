import os

import cv2
import numpy as np
from PIL import Image  # used for image processing

recognizer = (
    cv2.face.LBPHFaceRecognizer_create()
)  # Create the Local Binary Patterns Histograms (LBPH) face recognizer from our camera
path_for_imges = "dataSet/user"  # Path to the dataset of user images


def getImagesWithDb(path):
    imagePaths = [
        os.path.join(path, f) for f in os.listdir(path)
    ]  # Get the paths of all images in the dataset
    faceSamples = []  # List to hold the face samples
    ids = []  # List to hold the corresponding IDs

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert(
            "L"
        )  # "L" luminance mode - Convert the image to grayscale
        img_numpy = np.array(PIL_img, "uint8")  # Convert the image to a numpy array

        id = int(
            os.path.split(imagePath)[-1].split(".")[1]
        )  # Extract the ID from the filename

        faceSamples.append(img_numpy)  # Append the face sample to the list
        ids.append(id)  # Append the corresponding ID to the list

        cv2.imshow("Training", img_numpy)  # Display the image being processed
        cv2.waitKey(10)  # Wait for a short period before processing the next image

    return (
        np.array(ids),
        faceSamples,
    )  # Return the face samples and their corresponding IDs


ids, faces = getImagesWithDb(
    path_for_imges
)  # Get the face samples and IDs from the dataset

recognizer.train(faces, ids)  # Train the recognizer with the face samples and IDs
recognizer.save("recognizer/training_data.yml")  # Save the trained model to a file

cv2.destroyAllWindows()  # Close all OpenCV windows
