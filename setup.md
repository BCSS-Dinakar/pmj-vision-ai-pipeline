# 🛠️ SETUP GUIDE — Vision AI Pipeline

Full setup guide from zero to running, including folder structure, configuration, and training.

---

## ✅ STEP 1 — First Time Setup

### 1.1 — Create a Python Virtual Environment

```bash
python3 -m venv env
source env/bin/activate
```

> You will see `(env)` appear in your terminal prompt. This means the environment is active.

### 1.2 — Install All Required Packages

```bash
pip3 install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `opencv-python` | Reads RTSP camera streams and saves images |
| `ultralytics` | YOLOv8-Seg for person detection + segmentation |
| `numpy` | Fast array processing |
| `torch` | PyTorch deep learning engine |
| `torchvision` | Provides ResNet18 model and transforms |
| `Pillow` | Image loading for ResNet preprocessing |

---

## 📁 STEP 2 — Project Structure

```
Automation-modeltraining/
│
├── main.py                  ← Main script. Run this to start everything.
├── cameras.json             ← RTSP camera links per store
├── requirements.txt
├── README.md
├── setup.md                 ← This file
├── reference_data_guide.md
│
├── yolov8s-seg.pt           ← YOLOv8 segmentation model weights
├── reference_cache.pkl      ← Auto-generated. Delete to force embedding rebuild.
│
├── reference_data/          ← YOU PROVIDE — sample clothing photos per section
│   ├── sec1/
│   ├── sec2/
│   ├── sec3/ ... sec8/ sec10/
│   └── customers/
│
├── dataset/                 ← AUTO CREATED — raw camera output per day
│   └── YYYY-MM-DD/
│       └── <store-name>/
│           ├── images/              ← Good frames (person detected)
│           ├── blur/                ← Rejected low-quality frames
│           ├── annotations/
│           │   ├── images/          ← Frames with drawn boxes + labels
│           │   └── txt/             ← YOLO polygon label files
│           ├── crops/               ← Regular JPG crops per section label
│           └── segmented_crops/     ← Transparent RGBA PNG crops per section
│
└── training_dataset/        ← AUTO CREATED — final YOLO-ready dataset
    ├── data.yaml             ← Class names + split paths (auto-generated)
    ├── train/                ← 70% of data for training
    │   ├── images/
    │   └── labels/
    ├── val/                  ← 20% of data for checking accuracy
    │   ├── images/
    │   └── labels/
    └── test/                 ← 10% of data for final testing
        ├── images/
        └── labels/
```

---

## 📷 STEP 3 — Configure `cameras.json`

Edit this file with your RTSP links:

```json
[
    {
        "site_name": "your-store-name",
        "cameras": [
            {
                "camera_id": "CAM-01",
                "rtsp_url": "rtsp://username:password@ip:port/stream"
            }
        ]
    }
]
```

> You can add multiple stores and multiple cameras per store. All cameras run in parallel automatically.

---

## 🖼️ STEP 4 — Set Up `reference_data/`

This is the only manual step. Add sample clothing photos for each section so the AI knows what each section's uniform looks like.

**Rules:**
- Create one folder per section (must match `CLASS_MAPPING` names in `main.py`)
- Put **5–10 clear photos** of the uniform per section (minimum 3)
- Put **10–15 random casual clothing photos** in `customers/`
- Formats supported: `.jpg`, `.jpeg`, `.png`

> 📖 See [reference_data_guide.md](./reference_data_guide.md) for photo guidelines and tips.

**Important:** On first run, the script builds embeddings + color histograms for every reference image and saves them to `reference_cache.pkl`. On all future runs, this cache is loaded instantly — no reprocessing needed. If you add or change reference photos, **delete `reference_cache.pkl`** and re-run.

---

## 🚀 STEP 5 — Run the Script

```bash
python3 main.py
```

**What happens:**
```
1. Reads cameras.json → loads all camera links
2. Loads reference_cache.pkl (or builds it from reference_data/ if missing)
3. Launches one process per camera → all cameras run in parallel
4. Per frame:
   a. Quality check → blurry / dark / overexposed → saved to blur/ (skipped)
   b. YOLOv8-Seg detects people → segmentation masks + bounding boxes
   c. For each person: RGBA transparent crop built from their mask
   d. Crop matched against references: ResNet18 (60%) + HSV histogram (40%)
   e. Section label assigned; prints "Matched: sec2 Score: 0.812" to console
   f. YOLO polygon .txt label saved, annotated image saved, crops saved
5. All cameras finish → training_dataset/ is auto-built and split
```

**Console output:**
```
  📷  CAMERA CONNECTED  |  GF-37-CAM-01  — Starting capture...
  ✅  CLEAR  |  GF-37-CAM-01  [ 1/20]  |  Sharpness: 9800.1  Brightness: 94.2  Detail: 55.1
  Matched: sec2  Score: 0.812
  🚫  BLUR   |  GF-37-CAM-01  [ 2/20]  |  Sharpness:   45.2  Brightness:128.0  Detail:  0.4
  🚫  EMPTY  |  GF-37-CAM-01  [ 3/20]  |  Sharpness: 8200.0  Brightness: 90.0  Detail: 48.0
  ❌  CAMERA OFFLINE  |  GF-35-CAM-05  — Cannot connect.
  🏁  ALL CAMERAS FINISHED — MAX_IMAGES reached for every camera.
```

| Status | Meaning |
|---|---|
| `✅ CLEAR` | Good frame, person detected, saved |
| `🚫 BLUR` | Frame failed quality check, saved to `blur/` |
| `🚫 EMPTY` | Good frame but no person detected — nothing saved |
| `❌` | Camera offline or unreachable |

---

## ⚙️ STEP 6 — Configuration Settings

Edit these at the top of `main.py` under `# CONFIGURATION`:

| Setting | Default | Meaning |
|---|---|---|
| `MAX_IMAGES` | `20` | Frames to process per camera before stopping |
| `FRAME_INTERVAL` | `2` | Seconds between each captured frame |
| `BLUR_THRESHOLD` | `60` | Laplacian variance below this = rejected as blurry |

Matching settings (in `ReferenceMatcher.__init__`):

| Setting | Default | Meaning |
|---|---|---|
| `threshold` | `0.75` | Combined score needed to accept a section match (0 to 1) |

> **Tip:** Lower `threshold` (e.g. `0.60`) → more section matches, fewer `customers` labels. Higher → stricter matching.

---

## 📦 STEP 7 — Use the Training Dataset

After the script finishes, `training_dataset/` is ready. Train a new YOLOv8 model with:

```bash
yolo train model=yolov8s-seg.pt data=training_dataset/data.yaml epochs=50 imgsz=640
```

> Use `yolov8s-seg.pt` (not `yolov8s.pt`) since your labels are polygon segmentation format.

---

## 🔁 Running Again

Every time you run `python3 main.py`:
- New images are **added** to `dataset/` (organized by today's date)
- `training_dataset/` is **rebuilt fresh** from all collected data across all dates
- The more you run it, the larger and better your dataset grows

---

## ❓ Common Issues

| Problem | Fix |
|---|---|
| `❌ CAMERA OFFLINE` | Wrong RTSP link or camera is off — check `cameras.json` |
| All frames `🚫 BLUR` | Camera stream is initializing — wait and retry |
| Everything labeled `customers` | Lower `threshold` or add more/better reference photos |
| Wrong section predictions | Delete `reference_cache.pkl` then re-run to rebuild |
| `0 Labeled images` | No people visible — check camera angle |
| `ModuleNotFoundError` | Run `pip3 install -r requirements.txt` inside `env` |
