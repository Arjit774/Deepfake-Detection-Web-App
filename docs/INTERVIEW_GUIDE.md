# Interview Guide — Deepfake Detection Architecture

This guide is written so you can explain **this exact codebase** confidently in a placement interview. It covers the data flow, the shape of every tensor, why each design choice was made, and the questions an interviewer is likely to ask.

---

## 1. The 30-second pitch

> "It's a Flask web app that detects deepfakes in videos and images. OpenCV samples frames, InceptionV3 turns each frame into a 2048-dimensional feature vector, and a stacked GRU reads a sequence of up to 20 frames and outputs a sigmoid probability that the media is fake. For long videos I run multiple 20-frame windows and average their probabilities. The frontend uses Fetch API so inference happens asynchronously without reloading the page."

That is the whole system. Everything below is the detail you add when asked to go deeper.

---

## 2. End-to-end architecture

```
Browser (drag & drop)
        |
        |  multipart/form-data  (fetch POST /predict)
        v
Flask route: validate extension, size, decodability
        |
        +---------------------------+
        |                           |
     IMAGE                        VIDEO
        |                           |
  cv2.imdecode              cv2.VideoCapture
  centre-crop               seek to evenly spaced frames
  resize 224x224            centre-crop + resize 224x224
  BGR -> RGB                BGR -> RGB
        |                           |
        +-------------+-------------+
                      |
            list of RGB frames (N x 224 x 224 x 3)
                      |
            InceptionV3 (frozen, ImageNet)
            pooled average -> N x 2048
                      |
         20-frame windows with boolean mask
                      |
        GRU(64) -> BN -> GRU(32) -> BN -> GRU(16)
                -> Dropout -> Dense(32)
                -> Dropout -> Dense(1, sigmoid)
                      |
        P(fake) per window -> mean -> verdict + confidence
```

---

## 3. Frame extraction and video sampling

### 3.1 Why not process every frame?

A 30-second 30-fps video has 900 frames. Running InceptionV3 on all of them would be ~900 forward passes, which is slow and highly redundant: neighbouring frames are nearly identical. Sampling gives almost the same signal at a fraction of the cost. This is **temporal downsampling**, and it is standard in video understanding pipelines.

### 3.2 How OpenCV is used

```python
capture = cv2.VideoCapture(str(path))
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(capture.get(cv2.CAP_PROP_FPS))
```

- `CAP_PROP_FRAME_COUNT` gives the number of frames.
- `CAP_PROP_FPS` gives frames per second, so duration = `total_frames / fps`.

To pick frames evenly:

```python
indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int64)
for index in indices:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
```

`np.linspace` produces equally spaced indices across the whole clip. Seeking directly to a frame (`CAP_PROP_POS_FRAMES`) is much faster than reading sequentially, especially for long videos.

Some codecs/containers seek poorly. The code has a **fallback**: it resets to frame 0 and reads sequentially, keeping every `total_frames // max_frames`-th frame. This makes the pipeline robust across MP4, AVI, MOV, MKV, and WEBM.

### 3.3 Preprocessing each frame

```python
frame = crop_center_square(frame)          # square centre crop
frame = cv2.resize(frame, (224, 224))      # model input size
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

Three details matter:

1. **Centre crop** — matches the training notebook. Consistency between training and inference preprocessing is critical; changing it silently hurts accuracy.
2. **224×224** — InceptionV3's native input resolution.
3. **BGR → RGB** — OpenCV loads in BGR order, but InceptionV3 expects RGB. The original notebook used `frame[:, :, [2, 1, 0]]`, which is the same channel reversal.

### 3.4 Image path

Images skip the video loop entirely:

```python
buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8)
image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
frames = [prepare_frame(image)]
```

`cv2.imdecode` is used instead of `cv2.imread` because the upload is already on disk with an ASCII UUID name, and `imdecode` gives explicit control over failures (it returns `None` for corrupt data, which becomes a clean 422 error).

The image is treated as a **sequence of length 1**. The model already supports variable-length sequences through its mask, so this is the mathematically correct adaptation. A caveat worth mentioning: the GRU was trained mostly on 20-frame clips, so image accuracy is less validated than video accuracy. An alternative is to repeat the image 20 times; this is a reasonable experiment but changes the temporal distribution.

### 3.5 How many frames?

Default `MAX_VIDEO_FRAMES = 32`.

- 32 frames gives **two overlapping 20-frame windows**: frames 0–19 and 12–31.
- This covers more of the clip than the first 20 frames used in training, while staying CPU-friendly.
- The value is configurable via the `MAX_VIDEO_FRAMES` environment variable (1–64).

---

## 4. Feature extraction with InceptionV3

### 4.1 Why a separate feature extractor?

The `.h5` model does **not** take raw pixels. It takes 2048-D vectors produced by InceptionV3. The training notebook built the feature extractor separately and trained only the GRU head on cached features. At inference we must reproduce that exact contract.

```python
base = tf.keras.applications.InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3),
)
```

- `include_top=False` removes the 1000-class ImageNet classifier.
- `pooling="avg"` replaces the final spatial feature map with global average pooling, giving one **2048-D vector per frame**.
- `weights="imagenet"` means the convolutional filters already understand edges, textures, faces, and lighting. This is **transfer learning**.

### 4.2 Preprocessing is fused into the extractor

```python
inputs = tf.keras.Input((224, 224, 3))
preprocessed = tf.keras.applications.inception_v3.preprocess_input(inputs)
outputs = base(preprocessed)
return tf.keras.Model(inputs, outputs)
```

`inception_v3.preprocess_input` scales RGB pixels from `[0, 255]` to `[-1, 1]`. Because it is inside the model, the Flask code can pass raw RGB frames directly. This avoids the classic bug where training and serving use different normalization.

### 4.3 Batching for speed

```python
batch = np.asarray(frames, dtype=np.float32)       # (N, 224, 224, 3)
features = feature_extractor.predict(
    batch, batch_size=8, verbose=0
)                                                   # (N, 2048)
```

Passing the whole list at once is much faster than the original loop that called `predict()` once per frame. A batch size of 8 keeps memory bounded on CPU while still amortizing Python/Keras overhead.

### 4.4 Why InceptionV3?

- Strong ImageNet backbone with a good accuracy/compute trade-off.
- Already used in training — switching backbones at inference would invalidate the learned GRU weights.
- 2048-D pooled features are compact enough to cache and sequence.

If asked what you would change: EfficientNet-B0/B4 or a face-specific backbone (ArcFace) often improves deepfake detection, but you would have to **retrain the GRU head** because the feature dimension and distribution change.

---

## 5. Model inference: the GRU classifier

### 5.1 Input contract

From the training notebook:

```python
frame_features_input = keras.Input((20, 2048))   # float32
mask_input           = keras.Input((20,), dtype="bool")
```

At serving time:

```python
frame_features = np.zeros((1, 20, 2048), dtype=np.float32)
frame_mask     = np.zeros((1, 20), dtype=bool)

length = min(number_of_frames, 20)
frame_features[0, :length, :] = features[:length]
frame_mask[0, :length] = True

prediction = model.predict([frame_features, frame_mask], verbose=0)
```

Shape walk-through: **batch = 1**, **timesteps = 20**, **features = 2048**. The mask is boolean: `True` for a real frame, `False` for zero padding.

### 5.2 Why a mask?

Videos can be shorter than 20 frames (or an image is only 1 frame). Instead of requiring exactly 20 frames, we pad with zeros and tell the GRU which timesteps are valid. Masked timesteps do **not** update the GRU state, so the final hidden state is computed only from real frames. This is the same mechanism Keras uses for variable-length sequences in NLP.

### 5.3 Layer stack and what each layer does

```
Input (1, 20, 2048)
  -> GRU(64, return_sequences=True, masked)
  -> BatchNormalization
  -> GRU(32, return_sequences=True, masked)
  -> BatchNormalization
  -> GRU(16)                     # only final hidden state
  -> Dropout(0.5)
  -> Dense(32, relu)
  -> Dropout(0.5)
  -> Dense(1, sigmoid)
```

- **GRU (Gated Recurrent Unit)** — models temporal dependencies. Its update and reset gates learn which frame information to keep or forget. It has fewer parameters than an LSTM, which reduces overfitting on a small dataset.
- **Stacked GRUs** — the first layer learns short-range changes, deeper layers learn longer-range temporal patterns (e.g., inconsistent blinking or lip-sync artifacts).
- **BatchNormalization** — stabilises activations and speeds up training.
- **Dropout(0.5)** — regularisation; active only during training, automatically disabled at inference.
- **Dense(32, relu)** — non-linear combination of temporal features.
- **Dense(1, sigmoid)** — squashes the output to `[0, 1]`, interpreted as `P(fake)`.

### 5.4 Loss and label convention

Training used:

```python
labels = np.array(labels == "FAKE").astype(np.int)
model.compile(loss="binary_crossentropy", optimizer=Adam(1e-4))
```

So **label 1 = FAKE**, label 0 = REAL, and the sigmoid output is the probability of FAKE. This is why the threshold logic is:

```python
is_fake = fake_probability >= 0.5
```

### 5.5 Why sequence-level classification instead of per-frame?

A single frame can look real while the video is fake (or vice versa). Deepfakes often reveal themselves through **temporal inconsistencies** — flickering, boundary artifacts that move, unnatural blinking, mismatched head motion. A GRU can aggregate evidence across frames and make a more robust decision than a per-frame CNN.

This is also why the requirement "average frame predictions" needs care: the `.h5` model is **not** a per-frame classifier. It outputs one probability per **sequence window**. The code therefore averages **window predictions**, which is the correct interpretation for this architecture.

---

## 6. Aggregation: from windows to a final verdict

### 6.1 Single window

If the sampled feature matrix has ≤ 20 rows (images, short clips), one sequence is built with a mask and the model returns one probability.

### 6.2 Multiple windows

For clips with more than 20 sampled frames, the code creates 20-frame windows:

```python
starts = list(range(0, count - 20 + 1, 20))
tail_start = count - 20
if starts[-1] != tail_start:
    starts.append(tail_start)
```

With 32 frames: `[0, 12]` → windows `0–19` and `12–31`. The overlap ensures the end of the clip is represented. With 40 frames: `[0, 20]`, two non-overlapping windows. With 24 frames: `[0, 4]`, so the final four frames are still covered.

### 6.3 Averaging

```python
fake_probability = float(np.mean(window_scores))
```

- The mean reduces variance: one unusual window cannot dominate the verdict.
- We average **probabilities**, not logits, because the final score should be interpretable as a probability and the UI displays it directly.
- Alternatives: median (more robust to a single bad window), max (high-recall, flags any suspicious window), or a weighted average that prioritises the centre of the clip. Mention these as trade-offs.

### 6.4 Verdict and confidence

```python
is_fake = fake_probability >= DECISION_THRESHOLD   # default 0.5
verdict = "DEEPFAKE" if is_fake else "REAL"
confidence = fake_probability if is_fake else 1 - fake_probability
```

- `fake_probability` = model's belief that the media is fake.
- `confidence` = probability of the **predicted class**. A REAL verdict with `fake_probability = 0.12` has 88% confidence.
- The threshold is configurable. Lowering it increases recall for fake detection (fewer missed fakes) at the cost of more false positives. In a moderation context you might choose 0.35–0.45; in a legal context you might choose 0.7+.

### 6.5 Example

```
window 1 P(fake) = 0.91
window 2 P(fake) = 0.96
mean              = 0.935
verdict           = DEEPFAKE
confidence        = 93.5%
```

---

## 7. Video vs image: what changes

| Step | Video | Image |
| --- | --- | --- |
| Decode | `cv2.VideoCapture` | `cv2.imdecode` |
| Frames | up to 32 sampled | 1 |
| Sequence | 20-frame windows | 1-frame masked sequence |
| Aggregation | mean of window scores | single score |
| Metadata | fps, duration, total frames | width, height |

Everything after the frame list is identical, which is a nice architectural point: **the model is modality-agnostic once frames are produced**.

---

## 8. Backend engineering decisions

### 8.1 Lazy model loading

TensorFlow import + model load + InceptionV3 construction can take **1–2 minutes on a CPU-only machine** (TensorFlow import alone can be 30–60 seconds on Windows). Loading at module import would make every Flask worker slow to boot. The code uses **double-checked locking**:

```python
if _model is not None:
    return _model, _feature_extractor
with _model_lock:
    if _model is not None:
        return
    # load once
```

`warm_up()` is called at startup when running `python app.py`, so the first request is fast. Under Gunicorn, each worker loads lazily on its first request unless you warm up in a post-fork hook.

### 8.2 Inference lock

```python
with _inference_lock:
    features = extract_features(...)
    scores = predict_windows(...)
```

Keras inference is generally thread-safe, but serialising it prevents memory spikes when several users upload at once on a small CPU instance. With one Gunicorn worker and four threads, uploads are decoded in parallel but inference is serialised.

### 8.3 Upload safety

- **Extension whitelist** (`IMAGE_EXTENSIONS | VIDEO_EXTENSIONS`), not just MIME type — browsers and attackers can lie about MIME.
- **UUID filenames** — prevents path traversal and collisions; also avoids OpenCV's poor handling of non-ASCII paths on Windows.
- **Size cap** via `MAX_CONTENT_LENGTH` plus a post-save check.
- **Decodability check** — a file that passes the extension check but cannot be decoded returns 422.
- **`finally` cleanup** — the temp file is deleted even if inference fails.

### 8.4 Error taxonomy

| Status | When |
| --- | --- |
| 400 | Missing file, empty filename, empty file |
| 413 | Upload exceeds size cap |
| 415 | Extension not allowed |
| 422 | Corrupt/undecodable media, no frames |
| 503 | Model file missing or TensorFlow failed |
| 500 | Unexpected inference error |

The frontend reads the JSON `error` field and shows it in a styled alert, so users never see a raw stack trace.

---

## 9. Frontend engineering decisions

- **Drag-and-drop + file input** — `dragenter/dragover/dragleave/drop` with a depth counter to avoid flicker; keyboard accessible via `Enter`/`Space`.
- **Client-side validation** — extension, empty file, and 200 MB limit are checked before the network call to save a round trip.
- **Preview** — `URL.createObjectURL` for video/image; the URL is revoked when the file is removed to avoid memory leaks.
- **Fetch API + `FormData`** — no page reload. The spinner overlay appears immediately, and the UI stays responsive during inference.
- **Progressive status text** — "Loading the neural network…", "Extracting frames…", "Running inference…", "Aggregating predictions…" so a 30-second CPU run does not feel frozen.
- **Result card** — `data-verdict="fake|real"` drives the entire colour theme (red/green) through CSS, plus a confidence bar animated with `requestAnimationFrame` and a CSS transition.
- **`aria-live`** — screen readers announce the result; the overlay has `role="status"`.

---

## 10. Likely interview questions

### "Why did you use a GRU instead of an LSTM or a Transformer?"

GRUs have fewer gates (2 vs 3), so they train faster and overfit less on a modest dataset. The temporal patterns here are relatively short (20 frames), so the extra capacity of an LSTM is not clearly beneficial. A Transformer would need more data and compute; it is a good future improvement once you have a large labelled corpus.

### "Why 20 frames?"

It is a compute/context trade-off. Twenty frames capture roughly 0.7 seconds at 30 fps, enough to see blinking, lip movement, and boundary flicker. More frames increase the sequence length quadratically in attention models and linearly in GRU compute. It was also the training configuration, so inference must match it.

### "What happens if a video is shorter than 20 frames?"

The sequence is zero-padded to 20 and the boolean mask marks only the real frames. Masked timesteps do not update the GRU state, so the model behaves as if the sequence were shorter. For a single image, the mask has exactly one `True`.

### "What happens if a video is longer than 20 sampled frames?"

The features are split into multiple 20-frame windows, an end-aligned window is added if the last full window does not reach the final frame, and the per-window fake probabilities are averaged. This covers the whole clip rather than only the beginning.

### "How do you know the model is not just memorising faces?"

The InceptionV3 backbone is frozen and pretrained on ImageNet, and the GRU is trained on features. Still, dataset bias is a real risk: if the dataset has a shortcut (compression artifacts, actors, lighting), the model may learn it. Mitigations include diverse data, face-cropping, augmentation, group-wise evaluation, and testing on an external dataset.

### "How would you evaluate it?"

Accuracy alone is not enough. Use a confusion matrix, precision, recall, F1, ROC-AUC, and log loss. In deepfake detection, **false negatives** (missing a fake) and **false positives** (flagging a real video) have different costs, so choose the threshold based on the application. Also report calibration (reliability diagram / Brier score) because a raw sigmoid is often overconfident.

### "How would you improve accuracy?"

1. Face detection and alignment (MTCNN/RetinaFace) before feature extraction.
2. A face-specific backbone or EfficientNet.
3. Frame-level attention instead of a plain GRU.
4. Temporal consistency features (optical flow, blinking statistics).
5. Strong augmentation: compression, blur, noise, resizing, colour jitter.
6. Test-time augmentation across multiple frame windows.
7. Probability calibration (temperature scaling/Platt scaling).
8. Train on harder, more diverse data and evaluate cross-dataset.

### "How would you scale this?"

- Move inference to a GPU or a dedicated worker (Celery/RQ) and return a job ID; the frontend polls `/result/{id}`.
- Cache InceptionV3 features for repeated frames.
- Use one Gunicorn worker per GPU; scale horizontally with a queue.
- Store uploads in object storage with signed URLs instead of local disk.
- Add rate limiting and authentication.

### "Why not run the model in the browser?"

TensorFlow.js could work for a small model, but this model depends on InceptionV3 + a GRU and is large. Server-side inference keeps the model private, supports large videos, and avoids downloading tens of megabytes to every client.

### "What are the limitations?"

Compression, low light, occlusion, extreme poses, and out-of-distribution data can cause errors. The model is trained on a specific dataset and may not generalise. It produces a probability, not proof. Image detection is a one-frame adaptation of a video model. It should be used as a screening tool with human review.

### "What happens if two users upload at the same time?"

Flask handles requests on separate threads. Uploads and preprocessing can run in parallel, but inference is protected by `_inference_lock`, so predictions run one at a time. The second request waits (the UI spinner is still visible). This is intentional to avoid memory spikes; on a larger deployment you would use a queue and more workers/GPUs.

### "Why did you set Gunicorn to one worker?"

Each worker imports TensorFlow and loads its own copy of the model (~hundreds of MB of memory). On a small instance, multiple workers would exhaust RAM. One worker with multiple threads plus serialised inference is the safe default; scale horizontally when memory allows.

### "Why `compile=False` when loading the model?"

We only need the forward pass. The saved model's optimizer state and custom training configuration are unnecessary, and skipping compilation avoids errors from missing custom objects or version mismatches.

---

## 11. Whiteboard flow (memorise this)

```
Upload
  -> extension/size validation
  -> decode (VideoCapture / imdecode)
  -> sample frames (linspace seek)
  -> centre crop + resize 224 + RGB
  -> InceptionV3 batch -> (N, 2048)
  -> windows of 20 + boolean mask
  -> GRU stack -> sigmoid P(fake) per window
  -> mean -> threshold 0.5 -> DEEPFAKE / REAL
  -> confidence = max(p, 1-p)
```

Key numbers to quote:

- Frame tensor: `(N, 224, 224, 3)` float32 RGB.
- Feature tensor: `(N, 2048)`.
- Model inputs: `(1, 20, 2048)` + `(1, 20)` bool.
- Model output: scalar in `[0, 1]`.
- Default sample count: 32 frames → 2 windows.
- Default threshold: 0.5.

---

## 12. Two-minute demo script

1. "This is the upload dashboard. It accepts video or image via drag-and-drop."
2. Drop a video. "The preview is local — `URL.createObjectURL` — nothing is uploaded yet."
3. Click **Run Deepfake Analysis**. "The spinner overlay is driven by a Fetch request, so the page never reloads."
4. While waiting: "On the server, OpenCV samples 32 frames, InceptionV3 extracts 2048-D features, and the GRU scores two 20-frame windows."
5. Result appears red or green. "The confidence bar shows the probability of the predicted class; the cards show frames, windows, and latency."
6. Drop an image. "The same model runs with a one-frame masked sequence."
7. Close with the limitation: "It's a screening tool, not forensic proof."

---

## 13. Honest limitations to mention

- Dataset bias and cross-dataset generalisation.
- Compression/codec sensitivity.
- Image mode is less validated than video mode.
- Single-frame or very short clips give the GRU less temporal evidence.
- The sigmoid is not necessarily calibrated.
- Adversarial attacks and deliberately manipulated compression can evade detection.
- Ethics: false accusations, consent, and privacy. The app deletes uploads immediately and performs inference locally.

Mentioning limitations is a strength in interviews — it shows you understand the system's boundaries, not just its happy path.
