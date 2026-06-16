# 🎥 Vision AI Pipeline — Automated Dataset Generator

This tool connects to your security cameras, automatically finds people, figures out which store section they belong to, and builds a ready-to-use AI training dataset — all without any manual work.

---

## 🤔 What Does This Do?

In simple words, here is what happens when you run this script:

```
1. Script reads cameras.json         → Loads all your camera links
2. Connects to all cameras at once   → Runs every camera in parallel
3. Takes a photo every 2 seconds     → From each camera stream
4. Quality Check                     → Blurry / dark / overexposed = rejected to blur/ folder
5. YOLO finds people                 → Draws boxes around every person visible
6. ResNet checks their clothing      → Compares to your reference_data/ sample photos
7. Assigns a section label           → sec1, sec2, sec3, ... or customers
8. Saves annotated image + .txt file → YOLO format label with bounding box coordinates
9. Builds training_dataset/          → 70% train, 20% val, 10% test — automatically
```

---

## ✨ Features

| Feature | What it does |
|---|---|
| **Multi-Camera** | All cameras run at the same time using Python multiprocessing |
| **Quality Filter** | Rejects blurry, too dark, or too bright frames automatically |
| **Person Detection** | Uses `yolov8s.pt` (Small model) for accurate CCTV-grade detection |
| **Clothing Matcher** | Uses ResNet18 AI to identify which section a person belongs to |
| **Auto Labeling** | Draws green boxes + section names on every detected person |
| **Dataset Builder** | Packages everything into a YOLO-ready training dataset with data.yaml |

---

## 📁 Project Structure

```
Automation-modeltraining/
│
├── main.py              ← Main script. Run this to start everything.
├── cameras.json         ← List of all your camera RTSP links
├── requirements.txt     ← Python packages to install
├── README.md            ← This file
├── setup.md             ← Full step-by-step setup guide
│
├── reference_data/      ← YOU FILL THIS — sample clothing photos per section
│   ├── sec1/
│   ├── sec2/
│   └── customers/
│
├── dataset/             ← AUTO CREATED — raw camera output organized by date
│   └── 2026-06-16/
│       └── store-name/
│           ├── images/          ← Good quality frames
│           ├── blur/            ← Rejected blurry frames
│           ├── annotations/
│           │   ├── images/      ← Frames with boxes drawn on them
│           │   └── txt/         ← YOLO .txt label files
│           └── crops/           ← Cropped person photos by section
│
└── training_dataset/    ← AUTO CREATED — final clean dataset for training
    ├── data.yaml         ← YOLO config (class names + paths)
    ├── images/
    │   ├── train/        ← 70% of data
    │   ├── val/          ← 20% of data
    │   └── test/         ← 10% of data
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## ⚙️ Quick Setup

**Step 1 — Create a virtual environment:**
```bash
python3 -m venv env
source env/bin/activate
```

**Step 2 — Install dependencies:**
```bash
pip3 install -r requirements.txt
```

**Step 3 — Add your reference photos:**
Put 3–5 clear clothing photos for each section inside `reference_data/`:
```
reference_data/sec1/photo1.jpg
reference_data/sec2/photo1.jpg
reference_data/customers/photo1.jpg
```

**Step 4 — Run the script:**
```bash
python3 main.py
```

> 📖 See [setup.md](./setup.md) for the full detailed setup guide including camera configuration.

---

## 🔧 Settings (in `main.py`)

You can change these at the top of `main.py` under `# CONFIGURATION`:

| Setting | Default | Meaning |
|---|---|---|
| `MAX_IMAGES` | `10` | Photos to collect per camera |
| `FRAME_INTERVAL` | `2` | Seconds between each photo |
| `BLUR_THRESHOLD` | `150` | Sharpness limit (lower = stricter) |

---

## 📊 Console Output Guide

When running, the terminal will print one line per frame:

```
[STARTED] GF-37-CAM-01                           ← Camera connected
[CLEAR] GF-37-CAM-01 Blr:9800 Brt:108 Edg:55 1/10  ← Good frame saved
[BLUR] GF-37-CAM-01 Blr:50 Brt:128 Edg:0.3 2/10    ← Frame rejected
[ERROR] GF-35-CAM-05                              ← Camera offline
[DONE] All cameras reached MAX_IMAGES            ← All cameras finished
```

**Score meaning:**
- `Blr` → Sharpness score (higher = sharper)
- `Brt` → Brightness (40–220 is normal)
- `Edg` → Edge/detail density (above 5 is good)

---

## 🚀 Train Your Model

After running the script, your `training_dataset/` is ready. Train a new YOLO model with:

```bash
yolo train model=yolov8s.pt data=training_dataset/data.yaml epochs=50 imgsz=640
```

---

## ❓ Common Issues

| Problem | Fix |
|---|---|
| `[ERROR] Camera-ID` | Camera offline or wrong RTSP link in `cameras.json` |
| All frames are `[BLUR]` | Camera stream is initializing. Wait and try again. |
| `0 Annotated images` | No people visible, or `reference_data/` is empty |
| `ModuleNotFoundError` | Run `pip3 install -r requirements.txt` in the `env` |
