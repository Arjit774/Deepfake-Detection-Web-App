# Deepfake Detection — Video & Image

A Flask web application that detects deepfake manipulation in **videos and still images** using the existing Keras model at `model/deepfake_video_model.h5`.

The model is a **spatio-temporal GRU classifier**:

```
[frame_features: (batch, 20, 2048), frame_mask: (batch, 20)]
        -> GRU(64) -> GRU(32) -> GRU(16) -> Dense(32) -> Dense(1, sigmoid)
        -> P(fake)
```

Each frame is encoded with an InceptionV3 backbone into a 2048-D vector. The GRU then reads up to 20 frames as a sequence and returns the probability that the clip is fake.

## Features

- **Video detection** — OpenCV samples frames evenly across the clip, extracts InceptionV3 features in batches, and averages predictions across 20-frame windows.
- **Image detection** — the image is decoded and passed through the same model as a one-frame sequence.
- **Modern dark dashboard** — drag-and-drop upload, media preview, async fetch, animated progress bar, and distinct red/green verdict cards.
- **Robust error handling** — clear JSON errors for missing uploads, unsupported formats, oversized files, corrupt media, and model-loading failures.
- **Safe uploads** — extension whitelist, UUID filenames, size limits, and guaranteed temp-file cleanup.
- **No page reload** — the spinner overlay runs while inference happens in the background.

## Project structure

```
.
├── app.py                          # Flask backend + inference pipeline
├── Procfile                        # Production process definition
├── requirements.txt
├── model/
│   └── deepfake_video_model.h5     # Keras GRU classifier
├── static/
│   └── spinner.gif                 # Loading animation used by the overlay
├── templates/
│   └── index.html                  # Dark dashboard UI + vanilla JS
├── docs/
│   └── INTERVIEW_GUIDE.md          # Full technical interview walkthrough
├── uploads/                        # Temporary uploads (cleaned automatically)
└── README.md
```

## Quick start

> **Python version:** TensorFlow 2.13 requires **Python 3.10 or 3.11**. Python 3.12+ and 3.14 are not supported by this pinned TensorFlow version. Use the conda or `py -3.10` commands below.

### Option A — Conda (recommended, works on Windows/macOS/Linux)

```bash
conda create -n deepfakedetection python=3.10 -y
conda activate deepfakedetection
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### Option B — venv with Python 3.10

**Windows (PowerShell):**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

**macOS / Linux:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000/** in your browser.

### First-run notes

1. **Cold start is slow.** Importing TensorFlow, loading the Keras model, and building InceptionV3 can take **1–2 minutes on a CPU-only machine**. When you run `python app.py`, this happens during warm-up before the server starts. Under Gunicorn it happens on the first request.
2. InceptionV3 downloads its ImageNet weights (~92 MB) to `~/.keras/models/` the first time. Internet access is required once; later runs use the cache.
3. **Warm predictions are fast.** After warm-up, a short clip typically takes a few seconds on CPU; longer clips scale with the number of sampled frames. The UI overlay shows progress the whole time.
4. If you change Python environments, the TensorFlow import and InceptionV3 build repeat on the next cold start.

## API

### `POST /predict`

Multipart form-data with a `file` field (the legacy `video` field is also accepted).

```bash
curl -X POST http://127.0.0.1:5000/predict -F "file=@sample.mp4"
curl -X POST http://127.0.0.1:5000/predict -F "file=@sample.jpg"
```

Success response:

```json
{
  "result": "DEEPFAKE",
  "confidence": 93.42,
  "verdict": "DEEPFAKE",
  "confidence_ratio": 0.9342,
  "fake_probability": 0.9342,
  "threshold": 0.5,
  "media_type": "video",
  "frames_analyzed": 32,
  "windows_analyzed": 2,
  "window_scores": [0.91, 0.9584],
  "processing_time_ms": 8421.7,
  "filename": "sample.mp4",
  "media": {
    "width": 1080,
    "height": 1920,
    "total_frames": 420,
    "fps": 30.0,
    "duration_seconds": 14.0
  }
}
```

- `fake_probability` is the averaged sigmoid output (0–1).
- `confidence` is the probability of the predicted class as a percentage (`fake_probability` for DEEPFAKE, `1 - fake_probability` for REAL).
- `windows_analyzed` is 1 for images and short clips; longer clips produce multiple windows.

Error responses use `{"error": "...", "code": "..."}` with these status codes:

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `missing_file`, `empty_filename`, `empty_file` | No usable upload |
| 413 | `file_too_large` | Exceeds `MAX_CONTENT_LENGTH_MB` |
| 415 | `unsupported_format` | Extension is not in the whitelist |
| 422 | `invalid_media` | Corrupt/undecodable image or video |
| 503 | `model_unavailable` | Model file missing or TensorFlow failed to load |
| 500 | `inference_error` | Unexpected processing failure |

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_error": null,
  "max_video_frames": 32,
  "max_upload_mb": 200,
  "threshold": 0.5
}
```

`status` is `"standby"` until the first prediction loads the models (unless warm-up is enabled, which is the default when running `python app.py`).

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_VIDEO_FRAMES` | `32` | Frames sampled per video (1–64) |
| `DECISION_THRESHOLD` | `0.5` | Fake probability needed for a DEEPFAKE verdict (0.05–0.95) |
| `MAX_CONTENT_LENGTH_MB` | `200` | Maximum upload size in MB |
| `WARMUP_ON_START` | `1` | Load models at startup when running `python app.py` |
| `PORT` | `5000` | HTTP port |
| `FLASK_DEBUG` | `0` | Set to `1` for the Flask debug error page (never expose publicly) |
| `LOG_LEVEL` | `INFO` | Python logging level |

Example:

```powershell
$env:MAX_VIDEO_FRAMES = "48"
$env:DECISION_THRESHOLD = "0.55"
python app.py
```

## How it works (short version)

1. **Validate** the extension and decode the file.
2. **Sample** up to 32 frames evenly across a video with OpenCV; images become a single frame.
3. **Preprocess** each frame: centre-crop to a square, resize to 224×224, convert BGR → RGB.
4. **Encode** frames in batches with InceptionV3 (`include_top=False`, average pooling) → 2048-D vectors.
5. **Sequence**: pack up to 20 feature vectors into `(1, 20, 2048)` and build a boolean `(1, 20)` mask. Clips with more than 20 sampled frames produce one or more additional 20-frame windows.
6. **Classify**: the GRU outputs a sigmoid fake probability per window.
7. **Aggregate**: average the window probabilities, apply the threshold, and return the verdict plus confidence.

See [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) for the full technical walkthrough, likely interview questions, and improvement ideas.

## Production deployment

The included `Procfile` is tuned for a single worker because each worker loads its own TensorFlow model:

```
web: gunicorn app:app --workers 1 --threads 4 --timeout 300
```

Notes:

- Use **one worker** with multiple threads on small instances; scale horizontally only if memory allows.
- Keep the request timeout generous (model warm-up + frame extraction can take 30–60 s on CPU).
- Set `FLASK_DEBUG=0` and `WARMUP_ON_START=1` in production.
- `gunicorn` is intentionally not installed on Windows by `requirements.txt`; run `python app.py` locally or use a Linux container/Render/Heroku.
- For a Windows production server, install `waitress` and run `waitress-serve --port=5000 app:app`.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
Make sure the virtual environment is activated and you are on Python 3.10/3.11, then reinstall with `pip install -r requirements.txt`.

**TensorFlow install fails on Python 3.12/3.14**
The pinned TensorFlow 2.13 does not publish wheels for those versions. Create a Python 3.10 environment as shown above.

**Model fails to load / 503 response**
Confirm `model/deepfake_video_model.h5` exists. Check the Flask logs for the TensorFlow traceback. The model is loaded with `compile=False`, so no custom loss functions are required.

**InceptionV3 download error**
The first run needs internet access to fetch ImageNet weights. Make sure the machine can reach `storage.googleapis.com`, or pre-populate `~/.keras/models/`.

**413 File too large**
Raise `MAX_CONTENT_LENGTH_MB`, or compress/trim the video.

**Video opens but returns 422**
The codec may not be supported by your OpenCV build. Try re-encoding to H.264 MP4.

**`gunicorn` not recognized on Windows**
Gunicorn is Unix-only. Use `python app.py` or install `waitress`.

## Limitations

- This is a screening tool, **not forensic proof**. Always combine it with human review.
- The model was trained on the Kaggle Deepfake Detection Challenge sample data; accuracy on other datasets, codecs, and demographics may vary.
- Image detection reuses a video-trained model with a one-frame sequence, so it is less validated than video detection.
- Heavy compression, low light, occlusion, and extreme head poses can degrade accuracy.
- The model outputs a probability, not a calibrated legal-grade certainty score.
