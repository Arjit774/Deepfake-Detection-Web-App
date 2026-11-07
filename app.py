"""
Deepfake Detection — Flask backend
===================================

Supports both video and image uploads and runs inference with the existing
Keras model at ``model/deepfake_video_model.h5``.

Model contract (from the training notebook)
-------------------------------------------
    inputs : [frame_features: (batch, 20, 2048), frame_mask: (batch, 20)]
    output : sigmoid probability, 1 = FAKE / DEEPFAKE, 0 = REAL

Pipeline
--------
1. Validate the upload (extension + decodability).
2. Video : evenly sample frames with OpenCV -> center-crop -> resize 224x224.
   Image : decode the single image -> center-crop -> resize 224x224.
3. Batch the RGB frames through an InceptionV3 feature extractor (2048-D).
4. Build one or more 20-frame sequences and run the GRU model.
5. Average the per-window probabilities to produce the final score.

The model is loaded lazily and protected by a lock, so the web server can boot
without paying the TensorFlow startup cost on every worker immediately.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Keep TensorFlow's C++ logging quiet before it is imported (lazy import below).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Paths & model constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "deepfake_video_model.h5"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# How many frames to sample from a video. 32 gives two overlapping 20-frame
# windows (0-19 and 12-31), which covers more of the clip than the first 20
# frames used during training while staying CPU-friendly.
DEFAULT_MAX_VIDEO_FRAMES = 32
FEATURE_BATCH_SIZE = 8

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".flv",
    ".wmv",
}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read an integer env var, falling back safely on bad input."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


MAX_VIDEO_FRAMES = _env_int("MAX_VIDEO_FRAMES", DEFAULT_MAX_VIDEO_FRAMES, 1, 64)
DECISION_THRESHOLD = _env_float("DECISION_THRESHOLD", 0.5, 0.05, 0.95)
MAX_CONTENT_LENGTH_MB = _env_int("MAX_CONTENT_LENGTH_MB", 200, 1, 2048)
MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH_MB * 1024 * 1024
DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
PORT = _env_int("PORT", 5000, 1, 65535)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("deepfake")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    JSON_SORT_KEYS=False,
)

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
_model = None
_feature_extractor = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_model_error: str | None = None


def build_feature_extractor(tf):
    """InceptionV3 backbone that turns an RGB frame into 2048 features.

    ``preprocess_input`` is folded into the model, so callers pass raw RGB
    pixels in the 0-255 range (exactly what OpenCV gives us after BGR->RGB).
    """
    base = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = base(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")


def load_models():
    """Load the GRU classifier and the InceptionV3 feature extractor once."""
    global _model, _feature_extractor, _model_error

    if _model is not None and _feature_extractor is not None:
        return _model, _feature_extractor

    with _model_lock:
        # Double-checked locking: another request may have loaded them.
        if _model is not None and _feature_extractor is not None:
            return _model, _feature_extractor

        if not MODEL_PATH.is_file():
            _model_error = f"Model file not found at {MODEL_PATH}"
            raise FileNotFoundError(_model_error)

        try:
            import tensorflow as tf

            logger.info("Loading Keras model from %s", MODEL_PATH)
            model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            logger.info("Building InceptionV3 feature extractor (ImageNet weights)")
            feature_extractor = build_feature_extractor(tf)

            _model = model
            _feature_extractor = feature_extractor
            _model_error = None
            logger.info("Models loaded successfully.")
            return _model, _feature_extractor
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            _model_error = str(exc)
            logger.exception("Failed to load the model/feature extractor.")
            raise


def warm_up() -> None:
    """Optionally load models at boot so the first request is fast."""
    logger.info(
        "Warming up the model. The first run can take 1-2 minutes on CPU "
        "while TensorFlow imports and InceptionV3 is built."
    )
    try:
        load_models()
        logger.info("Warm-up complete.")
    except Exception:  # noqa: BLE001 - the health endpoint reports the error
        logger.warning("Warm-up failed; models will retry on the next request.")


# ---------------------------------------------------------------------------
# Frame / image preprocessing
# ---------------------------------------------------------------------------
class MediaDecodeError(Exception):
    """Raised when an upload cannot be decoded into usable frames."""


def crop_center_square(frame: np.ndarray) -> np.ndarray:
    """Match the training preprocessing: square centre crop."""
    height, width = frame.shape[:2]
    min_dim = min(height, width)
    start_x = (width // 2) - (min_dim // 2)
    start_y = (height // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def prepare_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> centre-cropped, 224x224 RGB uint8 frame."""
    frame = crop_center_square(frame_bgr)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_image_frames(path: Path) -> tuple[list[np.ndarray], dict]:
    """Decode a static image and return it as a one-frame sequence."""
    try:
        buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        raise MediaDecodeError("The image could not be read.") from exc

    if image is None:
        raise MediaDecodeError(
            "The image file is corrupt or uses an unsupported codec."
        )

    height, width = image.shape[:2]
    meta = {
        "width": int(width),
        "height": int(height),
        "total_frames": 1,
        "fps": None,
        "duration_seconds": None,
    }
    return [prepare_frame(image)], meta


def sample_video_frames(path: Path, max_frames: int) -> tuple[list[np.ndarray], dict]:
    """Sample frames evenly across a video using OpenCV.

    Prefer random-access seeking (fast for long clips) and fall back to a
    sequential scan if the container/codec does not support seeking.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise MediaDecodeError(
            "The video could not be opened. It may be corrupt or use an "
            "unsupported codec."
        )

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frames: list[np.ndarray] = []

        # --- Preferred path: seek to evenly spaced frame indices -----------
        if total_frames > 0:
            wanted = min(max_frames, total_frames)
            indices = np.unique(
                np.linspace(0, total_frames - 1, num=wanted, dtype=np.int64)
            )
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(prepare_frame(frame))

        # --- Fallback: sequential read, keeping every Nth frame ------------
        if not frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            step = max(1, total_frames // max_frames) if total_frames > 0 else 1
            index = 0
            while len(frames) < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % step == 0:
                    frames.append(prepare_frame(frame))
                index += 1

        if not frames:
            raise MediaDecodeError("No decodable frames were found in the video.")

        duration = round(total_frames / fps, 2) if fps > 0 and total_frames else None
        meta = {
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "fps": round(fps, 2) if fps else None,
            "duration_seconds": duration,
        }
        return frames, meta

    finally:
        capture.release()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def extract_frame_features(
    feature_extractor, frames: list[np.ndarray]
) -> np.ndarray:
    """Run all frames through InceptionV3 in batches -> (N, 2048)."""
    batch = np.asarray(frames, dtype=np.float32)
    features = feature_extractor.predict(
        batch, batch_size=FEATURE_BATCH_SIZE, verbose=0
    )
    return np.asarray(features, dtype=np.float32)


def sequence_windows(features: np.ndarray) -> list[np.ndarray]:
    """Split frame features into 20-frame windows for the GRU.

    Clips longer than 20 sampled frames produce additional windows. An
    end-aligned window is always added when the last full window does not
    already reach the final frame, so no part of the clip is ignored.
    """
    count = features.shape[0]
    if count <= MAX_SEQ_LENGTH:
        return [features]

    starts = list(range(0, count - MAX_SEQ_LENGTH + 1, MAX_SEQ_LENGTH))
    tail_start = count - MAX_SEQ_LENGTH
    if starts[-1] != tail_start:
        starts.append(tail_start)

    return [features[start : start + MAX_SEQ_LENGTH] for start in starts]


def predict_windows(model, features: np.ndarray) -> list[float]:
    """Return the sigmoid fake-probability for each sequence window."""
    scores: list[float] = []

    for window in sequence_windows(features):
        length = min(window.shape[0], MAX_SEQ_LENGTH)

        # The model expects (1, 20, 2048) features and a (1, 20) boolean mask.
        frame_features = np.zeros(
            (1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32
        )
        frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype=bool)

        frame_features[0, :length, :] = window[:length]
        frame_mask[0, :length] = True

        prediction = model.predict([frame_features, frame_mask], verbose=0)
        scores.append(float(np.asarray(prediction).reshape(-1)[0]))

    return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def error_response(message: str, status: int, code: str):
    return jsonify({"error": message, "code": code}), status


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_exc):
    return error_response(
        f"File is too large. The maximum upload size is {MAX_CONTENT_LENGTH_MB} MB.",
        413,
        "file_too_large",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if _model is not None else "standby",
            "model_loaded": _model is not None,
            "model_error": _model_error,
            "max_video_frames": MAX_VIDEO_FRAMES,
            "max_upload_mb": MAX_CONTENT_LENGTH_MB,
            "threshold": DECISION_THRESHOLD,
        }
    )


@app.post("/predict")
def predict():
    started = time.perf_counter()

    # ``file`` is the new field name; ``video`` keeps the old API working.
    # Fall back to ``video`` when ``file`` is present but has an empty name.
    upload = request.files.get("file")
    if upload is None or not (upload.filename or "").strip():
        upload = request.files.get("video")

    if upload is None:
        return error_response(
            "No file was uploaded. Send a multipart/form-data request with a "
            "'file' field.",
            400,
            "missing_file",
        )

    original_name = (upload.filename or "").strip()
    if not original_name:
        return error_response("The uploaded file has no filename.", 400, "empty_filename")

    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return error_response(
            f"Unsupported file type '{extension or 'unknown'}'. Allowed types: {allowed}.",
            415,
            "unsupported_format",
        )

    media_type = "image" if extension in IMAGE_EXTENSIONS else "video"

    # Store with an ASCII UUID name: avoids path traversal, collisions, and
    # OpenCV's poor handling of non-ASCII paths on Windows.
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"

    try:
        upload.save(str(temp_path))
        if temp_path.stat().st_size == 0:
            return error_response("The uploaded file is empty.", 400, "empty_file")

        if temp_path.stat().st_size > MAX_CONTENT_LENGTH:
            return error_response(
                f"File is too large. The maximum upload size is "
                f"{MAX_CONTENT_LENGTH_MB} MB.",
                413,
                "file_too_large",
            )

        # Decode before loading TensorFlow so corrupt uploads fail fast with
        # a 422 instead of paying a 1-2 minute cold start.
        if media_type == "image":
            frames, media_meta = load_image_frames(temp_path)
        else:
            frames, media_meta = sample_video_frames(temp_path, MAX_VIDEO_FRAMES)

        # Load the models only once the media is known to be decodable.
        try:
            model, feature_extractor = load_models()
        except FileNotFoundError as exc:
            return error_response(
                f"Model file is missing: {exc}",
                503,
                "model_unavailable",
            )
        except Exception:  # noqa: BLE001
            return error_response(
                "The AI model could not be loaded. Check the server logs for details.",
                503,
                "model_unavailable",
            )

        # Serialise GPU/CPU inference: Keras inference is thread-safe in
        # theory, but one-at-a-time avoids memory spikes on small deployments.
        with _inference_lock:
            features = extract_frame_features(feature_extractor, frames)
            window_scores = predict_windows(model, features)

        fake_probability = float(np.mean(window_scores))
        is_fake = fake_probability >= DECISION_THRESHOLD
        verdict = "DEEPFAKE" if is_fake else "REAL"

        # Confidence = probability of the predicted class, as a percentage.
        confidence = fake_probability if is_fake else 1.0 - fake_probability
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        logger.info(
            "Prediction: %s (fake=%.4f) | type=%s frames=%d windows=%d in %.0f ms",
            verdict,
            fake_probability,
            media_type,
            features.shape[0],
            len(window_scores),
            elapsed_ms,
        )

        return jsonify(
            {
                # Primary fields consumed by the dashboard.
                "result": verdict,
                "confidence": round(confidence * 100, 2),
                # Extra context for the result card and API clients.
                "verdict": verdict,
                "confidence_ratio": round(confidence, 4),
                "fake_probability": round(fake_probability, 4),
                "threshold": DECISION_THRESHOLD,
                "media_type": media_type,
                "frames_analyzed": int(features.shape[0]),
                "windows_analyzed": len(window_scores),
                "window_scores": [round(score, 4) for score in window_scores],
                "processing_time_ms": elapsed_ms,
                "filename": secure_filename(original_name) or f"upload{extension}",
                "media": media_meta,
            }
        )

    except MediaDecodeError as exc:
        return error_response(str(exc), 422, "invalid_media")
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while processing the upload.")
        return error_response(
            "Something went wrong during analysis. Please try another file.",
            500,
            "inference_error",
        )
    finally:
        # Always clean up, even when validation or inference fails.
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary file: %s", temp_path)


if __name__ == "__main__":
    if os.getenv("WARMUP_ON_START", "1").lower() in {"1", "true", "yes", "on"}:
        warm_up()

    # The reloader would load TensorFlow twice; keep it off by default.
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG, use_reloader=False)
