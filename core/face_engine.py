from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from core.config import (
    DEFAULT_DISTANCE_THRESHOLD,
    DETECT_MIN_NEIGHBORS,
    DETECT_MIN_SIZE,
    DETECT_SCALE_FACTOR,
    FACE_SIZE,
    LOCAL_CASCADE_PATH,
    MODEL_PATH,
    USER_IMAGES_DIR,
    ensure_directories,
    logger,
)
from core.database import DatabaseManager, PersonProfile


@dataclass
class DetectedPersonResult:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    is_known: bool
    person_id: Optional[int]
    profile: Optional[PersonProfile]
    distance: float
    similarity_pct: float


def load_face_cascade() -> cv2.CascadeClassifier:
    """Load the Haar cascade classifier with fallback to OpenCV's built-in data path."""
    if LOCAL_CASCADE_PATH.exists():
        cascade = cv2.CascadeClassifier(str(LOCAL_CASCADE_PATH))
        if not cascade.empty():
            return cascade

    fallback_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(fallback_path))
    if cascade.empty():
        raise RuntimeError(
            f"Unable to load Haar cascade classifier from '{LOCAL_CASCADE_PATH}' or '{fallback_path}'."
        )
    return cascade


def create_lbph_recognizer():
    """Create an OpenCV LBPHFaceRecognizer, raising a clear error if opencv-contrib-python is missing."""
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "cv2.face.LBPHFaceRecognizer_create is unavailable. "
            "Please install 'opencv-contrib-python': pip install --upgrade --force-reinstall opencv-contrib-python"
        )
    return cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


def preprocess_face(gray_roi: np.ndarray, apply_equalization: bool = True) -> np.ndarray:
    """Standardize face crop size and lighting for higher LBPH accuracy."""
    resized = cv2.resize(gray_roi, FACE_SIZE, interpolation=cv2.INTER_AREA)
    if apply_equalization:
        return cv2.equalizeHist(resized)
    return resized


def distance_to_similarity(distance: float, threshold: float = DEFAULT_DISTANCE_THRESHOLD) -> float:
    """
    Convert OpenCV LBPH Chi-Square distance (0 = identical, lower = better) into an intuitive 0-100% similarity score.
    - Distance 0 -> 100%
    - Distance == threshold -> ~60% (minimum passing match score)
    - Distance >= 140 -> 0%
    """
    if distance <= 0:
        return 100.0
    if distance <= threshold:
        # Linearly map [0, threshold] -> [100%, 60%]
        return max(60.0, 100.0 - (distance / max(threshold, 1.0)) * 40.0)
    # Map (threshold, 140] -> (60%, 0%]
    span = max(140.0 - threshold, 1.0)
    score = 60.0 * (1.0 - (distance - threshold) / span)
    return float(max(0.0, min(100.0, score)))


class FaceRecognitionEngine:
    """
    Enterprise Face Detection, Enrollment, Training & Simultaneous Multi-Person Recognition Engine.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        ensure_directories()
        self.db = db_manager or DatabaseManager()
        self.face_cascade = load_face_cascade()
        self.recognizer = create_lbph_recognizer()
        self.model_loaded = False
        self.distance_threshold = DEFAULT_DISTANCE_THRESHOLD
        self.apply_equalization = True
        self.load_model_if_exists()

    def load_model_if_exists(self, model_path: Path = MODEL_PATH) -> bool:
        """Load the trained LBPH model if it exists on disk."""
        if model_path.exists() and model_path.stat().st_size > 0:
            try:
                self.recognizer.read(str(model_path))
                self.model_loaded = True
                logger.info("Loaded trained LBPH model from %s", model_path)
                return True
            except cv2.error as exc:
                logger.warning("Failed to load model from %s: %s", model_path, exc)
                self.model_loaded = False
                return False
        self.model_loaded = False
        return False

    def detect_faces(self, gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces simultaneously in a grayscale frame.
        Returns a list of (x, y, w, h) bounding boxes sorted largest-first.
        """
        raw_faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=DETECT_SCALE_FACTOR,
            minNeighbors=DETECT_MIN_NEIGHBORS,
            minSize=DETECT_MIN_SIZE,
        )
        if len(raw_faces) == 0:
            return []

        boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in raw_faces]
        # Sort by area descending so primary/closest faces come first
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes

    def save_enrollment_sample(
        self,
        person_id: int,
        sample_number: int,
        gray_frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Path:
        """Crop, normalize, and save a user's face image sample."""
        ensure_directories()
        x, y, w, h = bbox
        roi = gray_frame[y : y + h, x : x + w]
        processed = preprocess_face(roi, apply_equalization=self.apply_equalization)
        file_path = USER_IMAGES_DIR / f"User.{person_id}.{sample_number}.jpg"
        success = cv2.imwrite(str(file_path), processed)
        if not success:
            raise IOError(f"Failed to write sample image to {file_path}")
        return file_path

    def count_user_samples(self) -> Dict[int, int]:
        """Count how many image samples exist per user ID in dataSet/user."""
        counts: Dict[int, int] = {}
        if not USER_IMAGES_DIR.exists():
            return counts
        for img_path in USER_IMAGES_DIR.iterdir():
            if not img_path.is_file():
                continue
            parts = img_path.name.split(".")
            if len(parts) >= 3 and parts[0].lower() == "user":
                try:
                    uid = int(parts[1])
                    counts[uid] = counts.get(uid, 0) + 1
                except ValueError:
                    continue
        return counts

    def delete_user_samples(self, person_id: int) -> int:
        """Delete all saved dataset images for a given user ID."""
        removed = 0
        if not USER_IMAGES_DIR.exists():
            return removed
        for img_path in USER_IMAGES_DIR.iterdir():
            if not img_path.is_file():
                continue
            parts = img_path.name.split(".")
            if len(parts) >= 3 and parts[0].lower() == "user":
                try:
                    if int(parts[1]) == person_id:
                        img_path.unlink(missing_ok=True)
                        removed += 1
                except ValueError:
                    continue
        return removed

    def train_model(
        self,
        dataset_dir: Path = USER_IMAGES_DIR,
        model_path: Path = MODEL_PATH,
        progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    ) -> Tuple[int, int]:
        """
        Train the LBPH face recognizer on all valid images in `dataSet/user`.
        Returns (num_images_trained, num_unique_users).
        """
        ensure_directories()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_files = sorted(
            [
                p
                for p in dataset_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_extensions
            ]
        )

        face_samples: List[np.ndarray] = []
        ids: List[int] = []
        total = len(all_files)

        for idx, image_path in enumerate(all_files, start=1):
            parts = image_path.name.split(".")
            # Expected format: User.<id>.<sampleNum>.jpg
            if len(parts) < 3 or parts[0].lower() != "user":
                logger.warning("Skipping non-conforming file in dataset: %s", image_path.name)
                continue

            try:
                user_id = int(parts[1])
            except ValueError:
                logger.warning("Skipping file with non-integer ID: %s", image_path.name)
                continue

            try:
                with Image.open(image_path) as pil_img:
                    gray_pil = pil_img.convert("L")
                    img_numpy = np.array(gray_pil, "uint8")
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", image_path.name, exc)
                continue

            # Standardize dimensions for training consistency
            processed = preprocess_face(img_numpy, apply_equalization=self.apply_equalization)
            face_samples.append(processed)
            ids.append(user_id)

            if progress_callback is not None:
                progress_callback(idx, total, image_path)

        if not face_samples:
            raise ValueError(
                f"No valid training images found in '{dataset_dir}'. "
                "Please enroll at least one user before training."
            )

        self.recognizer = create_lbph_recognizer()
        self.recognizer.train(face_samples, np.array(ids, dtype=np.int32))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.recognizer.save(str(model_path))
        self.model_loaded = True

        unique_users = len(set(ids))
        logger.info(
            "Trained LBPH model on %d images across %d unique user(s). Saved to %s",
            len(face_samples),
            unique_users,
            model_path,
        )
        return len(face_samples), unique_users

    def recognize_faces_in_frame(
        self, gray_frame: np.ndarray, record_attendance: bool = True
    ) -> List[DetectedPersonResult]:
        """
        Simultaneously detect and recognize ALL people present in the frame.
        Uses in-memory profile caching for zero database latency per frame.
        """
        bboxes = self.detect_faces(gray_frame)
        results: List[DetectedPersonResult] = []

        for x, y, w, h in bboxes:
            if not self.model_loaded:
                results.append(
                    DetectedPersonResult(
                        bbox=(x, y, w, h),
                        is_known=False,
                        person_id=None,
                        profile=None,
                        distance=999.0,
                        similarity_pct=0.0,
                    )
                )
                continue

            roi = gray_frame[y : y + h, x : x + w]
            processed_roi = preprocess_face(roi, apply_equalization=self.apply_equalization)

            try:
                predicted_id, distance = self.recognizer.predict(processed_roi)
            except cv2.error:
                results.append(
                    DetectedPersonResult(
                        bbox=(x, y, w, h),
                        is_known=False,
                        person_id=None,
                        profile=None,
                        distance=999.0,
                        similarity_pct=0.0,
                    )
                )
                continue

            similarity_pct = distance_to_similarity(distance, self.distance_threshold)
            profile = self.db.get_person(int(predicted_id))

            # A valid match requires BOTH distance < threshold AND an existing profile in the database
            is_known = (distance < self.distance_threshold) and (profile is not None)

            if is_known and profile is not None and record_attendance:
                self.db.record_recognition(profile, similarity_pct, distance)

            results.append(
                DetectedPersonResult(
                    bbox=(x, y, w, h),
                    is_known=is_known,
                    person_id=int(predicted_id) if is_known else None,
                    profile=profile if is_known else None,
                    distance=float(distance),
                    similarity_pct=similarity_pct,
                )
            )

        return results


def draw_enterprise_hud(
    frame: np.ndarray,
    detections: List[DetectedPersonResult],
    fps: float = 0.0,
    model_loaded: bool = True,
    enrollment_progress: Optional[Tuple[int, int, str]] = None,
) -> np.ndarray:
    """
    Render modern HUD overlays onto a BGR video frame for simultaneous multi-person tracking.
    """
    annotated = frame.copy()
    frame_h, frame_w = annotated.shape[:2]

    # Top status bar overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 42), (18, 22, 31), -1)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

    known_count = sum(1 for d in detections if d.is_known)
    unknown_count = len(detections) - known_count

    status_text = (
        f"FACES: {len(detections)}  |  VERIFIED: {known_count}  |  UNKNOWN: {unknown_count}"
    )
    if not model_loaded and enrollment_progress is None:
        status_text += "  |  [MODEL NOT TRAINED]"

    cv2.putText(
        annotated,
        status_text,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 244, 248),
        1,
        cv2.LINE_AA,
    )

    if fps > 0:
        fps_text = f"{fps:.1f} FPS"
        cv2.putText(
            annotated,
            fps_text,
            (frame_w - 105, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (100, 220, 140),
            1,
            cv2.LINE_AA,
        )

    # Draw each detected person's bounding box & badge
    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det.bbox
        if det.is_known and det.profile is not None:
            color = (80, 215, 110)  # Emerald green (BGR)
            title = f"{det.profile.name} (ID: {det.profile.id})"
            subtitle = f"Age: {det.profile.age} | {det.profile.department} | {det.similarity_pct:.0f}%"
        else:
            color = (70, 85, 235)  # Crimson red (BGR)
            title = f"Unknown #{idx}"
            subtitle = f"Unverified (Dist: {det.distance:.1f})" if model_loaded else "Untrained Model"

        # Subtle bounding box + bold corner brackets
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)
        corner_len = max(14, min(w, h) // 5)
        thickness = 3
        # Top-left
        cv2.line(annotated, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(annotated, (x, y), (x, y + corner_len), color, thickness)
        # Top-right
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color, thickness)
        # Bottom-left
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color, thickness)
        # Bottom-right
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        # Label backdrop below or above box
        label_w = max(w, 220)
        label_y1 = y + h + 4 if (y + h + 52 < frame_h) else max(44, y - 52)
        label_y2 = label_y1 + 46

        sub_overlay = annotated.copy()
        cv2.rectangle(
            sub_overlay,
            (x, label_y1),
            (min(frame_w - 4, x + label_w), label_y2),
            (20, 25, 35),
            -1,
        )
        cv2.addWeighted(sub_overlay, 0.80, annotated, 0.20, 0, annotated)
        cv2.rectangle(
            annotated,
            (x, label_y1),
            (min(frame_w - 4, x + label_w), label_y2),
            color,
            1,
        )

        cv2.putText(
            annotated,
            title,
            (x + 8, label_y1 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            subtitle,
            (x + 8, label_y1 + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 210, 225),
            1,
            cv2.LINE_AA,
        )

    # Enrollment progress bar overlay if active
    if enrollment_progress is not None:
        current_sample, target_samples, user_name = enrollment_progress
        bar_y = frame_h - 46
        cv2.rectangle(annotated, (0, bar_y), (frame_w, frame_h), (18, 22, 31), -1)
        pct = min(1.0, current_sample / max(target_samples, 1))
        fill_w = int((frame_w - 28) * pct)
        cv2.rectangle(
            annotated, (14, bar_y + 24), (frame_w - 14, frame_h - 8), (50, 60, 75), -1
        )
        cv2.rectangle(
            annotated, (14, bar_y + 24), (14 + fill_w, frame_h - 8), (235, 160, 50), -1
        )
        cv2.putText(
            annotated,
            f"ENROLLING {user_name.upper()}: {current_sample}/{target_samples} SAMPLES CAPTURED",
            (14, bar_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 215, 100),
            1,
            cv2.LINE_AA,
        )

    return annotated

