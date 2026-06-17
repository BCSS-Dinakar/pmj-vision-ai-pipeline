# 🎥 Vision AI Pipeline — Automated Dataset Generator

Connects to your RTSP security cameras, detects people using YOLOv8-Seg, matches their section by clothing, and builds a ready-to-use YOLO training dataset — fully automated.

---

## 🤔 What Does This Do?

```
1. Read cameras.json          → Loads all camera RTSP links
2. Connect in parallel        → Each camera runs in its own process
3. Capture a frame every 2s   → From each RTSP stream
4. Quality check              → Blurry / dark / overexposed → saved to blur/ (not annotated)
5. YOLOv8-Seg detects people  → Draws segmentation masks + bounding boxes
6. Clothing matching          → ResNet18 embedding (60%) + HSV color histogram (40%)
                                compared against reference_data/ photos
7. Section label assigned     → sec1, sec2, ..., customers
8. Save outputs               → YOLO polygon .txt label + annotated image + transparent PNG crop
9. Build training_dataset/    → 70% train, 20% val, 10% test split — auto generated
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Multi-Camera** | All cameras run simultaneously via Python multiprocessing |
| **Quality Filter** | Rejects blurry (`BLUR_THRESHOLD`), too dark (< 40), too bright (> 220), or blank frames |
| **Person Detection** | YOLOv8s-Seg with `conf=0.5` and `retina_masks=True` for high-quality polygon masks |
| **Clothing Matcher** | ResNet18 (shape/texture) + HSV histogram (color) — threshold `0.75` |
| **Reference Cache** | Embeddings saved to `reference_cache.pkl` — fast startup on subsequent runs |
| **RGBA Crops** | Segmented person cutouts saved as transparent PNGs (background fully removed) |
| **Lazy Folders** | Output folders are only created when actual data needs to be saved |
| **Auto Dataset** | Packages everything into YOLO-ready `train/val/test` split with `data.yaml` |

---

## 📁 Project Structure

```
Automation-modeltraining/
│
├── main.py                  ← Main pipeline script
├── cameras.json             ← RTSP camera links per store
├── reference_data/          ← Clothing sample photos per section (you provide)
│   ├── sec1/
│   ├── sec2/
│   └── customers/
├── reference_cache.pkl      ← Auto-generated embedding cache (delete to rebuild)
├── yolov8s-seg.pt           ← YOLOv8 segmentation model weights
├── requirements.txt
├── README.md
├── setup.md
├── reference_data_guide.md
│
├── dataset/                 ← AUTO CREATED — raw daily output per store
│   └── YYYY-MM-DD/
│       └── store-name/
│           ├── images/              ← Good frames (person detected)
│           ├── blur/                ← Rejected low-quality frames
│           ├── annotations/
│           │   ├── images/          ← Annotated frames with drawn boxes
│           │   └── txt/             ← YOLO polygon label files
│           ├── crops/               ← Regular JPG crops per section
│           └── segmented_crops/     ← Transparent PNG crops per section
│
├── training_dataset/        ← AUTO CREATED — final YOLO-ready dataset
│   ├── data.yaml
│   ├── train/               ← 70% of data
│   │   ├── images/
│   │   └── labels/
│   ├── val/                 ← 20% of data
│   │   ├── images/
│   │   └── labels/
│   └── test/                ← 10% of data
│       ├── images/
│       └── labels/
│
└── dataset_3_by_vedic/      ← Standalone tools for reference dataset preparation
    ├── crop_images.py       ← Step 1: Extract crops from annotated images
    ├── clean_dataset.py     ← Step 2: Quality filter + diversity selection
    └── segment_dataset.py   ← Step 3: YOLO segmentation → transparent RGBA PNGs
```

---

## ⚙️ Quick Setup

**Step 1 — Virtual environment:**
```bash
python3 -m venv env
source env/bin/activate
```

**Step 2 — Install dependencies:**
```bash
pip3 install -r requirements.txt
```

**Step 3 — Add reference photos:**
Put **5–10 clear clothing photos** per section into `reference_data/`:
```
reference_data/sec1/photo1.jpg
reference_data/sec2/photo1.jpg
reference_data/customers/photo1.jpg
```
> 📖 See [reference_data_guide.md](./reference_data_guide.md) for photo guidelines.

**Step 4 — Add your cameras:**
Edit `cameras.json` with your RTSP links.

**Step 5 — Run:**
```bash
python3 main.py
```

---

## 🔧 Settings (top of `main.py`)

| Setting | Default | Meaning |
|---|---|---|
| `MAX_IMAGES` | `20` | Frames to process per camera |
| `FRAME_INTERVAL` | `2` | Seconds between each captured frame |
| `BLUR_THRESHOLD` | `60` | Laplacian variance below this = rejected as blurry |

Matcher settings (in `ReferenceMatcher.__init__`):

| Setting | Default | Meaning |
|---|---|---|
| `threshold` | `0.75` | Minimum combined score to accept a section match |

---

## 📊 Console Output

```
  📷  CAMERA CONNECTED  |  GF-37-CAM-01  — Starting capture...
  ✅  CLEAR  |  GF-37-CAM-01  [ 1/20]  |  Sharpness: 9800.1  Brightness: 94.2  Detail: 55.1
  Matched: sec2  Score: 0.812
  🚫  BLUR   |  GF-37-CAM-01  [ 2/20]  |  Sharpness:   45.2  Brightness:128.0  Detail:  0.4
  🚫  EMPTY  |  GF-37-CAM-01  [ 3/20]  |  Sharpness: 8200.0  Brightness: 90.0  Detail: 48.0
  ❌  CAMERA OFFLINE  |  GF-35-CAM-05  — Cannot connect.
  🏁  ALL CAMERAS FINISHED — MAX_IMAGES reached for every camera.
```

**Status icons:**
- `✅ CLEAR` — Good frame with at least one person detected and saved
- `🚫 BLUR`  — Frame failed quality check (saved to blur/ for review)
- `🚫 EMPTY` — Good frame but no person detected (nothing saved)
- `❌`       — Camera offline or unreachable

**Score columns:**
- `Sharpness` — Laplacian variance (rejected if < `BLUR_THRESHOLD`)
- `Brightness` — Mean pixel brightness (rejected if < 40 or > 220)
- `Detail`     — Canny edge density (rejected if < 5 — blank frame)

---

## 🚀 Train Your Model

After running, `training_dataset/` is ready. Train with:
```bash
yolo train model=yolov8s.pt data=training_dataset/data.yaml epochs=50 imgsz=640
```

---

## ❓ Common Issues

| Problem | Fix |
|---|---|
| `❌ CAMERA OFFLINE` | Wrong RTSP link or camera is off |
| All frames `🚫 BLUR` | Camera stream is initializing — wait and retry |
| `0 Labeled images` | No people visible, or `reference_data/` is empty |
| Everything labeled `customers` | Raise `threshold` or add more reference photos |
| `ModuleNotFoundError` | Run `pip3 install -r requirements.txt` in env |
| Wrong section predictions | Delete `reference_cache.pkl` and re-run to rebuild embeddings |
