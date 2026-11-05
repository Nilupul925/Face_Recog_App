from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image  # used for image opening and processing

from core.config import MODEL_PATH, USER_IMAGES_DIR, ensure_directories
from core.face_engine import FaceRecognitionEngine, preprocess_face

path_for_images = USER_IMAGES_DIR  # Path to the dataset of user images


def getImagesWithDb(path: Path = USER_IMAGES_DIR) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Load, validate, and preprocess all user face images for LBPH training."""
    ensure_directories()
    path = Path(path)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    )

    face_samples: List[np.ndarray] = []  # List to hold the face samples
    ids: List[int] = []  # List to hold the corresponding IDs

    for image_path in image_paths:
        parts = image_path.name.split(".")
        if len(parts) < 3 or parts[0].lower() != "user":
            continue

        try:
            user_id = int(parts[1])  # Extract the integer ID from User.<id>.<sample>.jpg
        except ValueError:
            continue

        with Image.open(image_path) as pil_img:
            gray_img = pil_img.convert("L")  # "L" luminance mode - Convert the image to grayscale
            img_numpy = np.array(gray_img, "uint8")  # Convert the image to a numpy array

        processed = preprocess_face(img_numpy, apply_equalization=True)
        face_samples.append(processed)
        ids.append(user_id)

    return np.array(ids, dtype=np.int32), face_samples


def main():
    ensure_directories()
    engine = FaceRecognitionEngine()
    num_images, num_users = engine.train_model(
        dataset_dir=USER_IMAGES_DIR,
        model_path=MODEL_PATH,
    )
    cv2.destroyAllWindows()
    print(
        f"Training complete: {num_images} face images across {num_users} user(s) -> {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
