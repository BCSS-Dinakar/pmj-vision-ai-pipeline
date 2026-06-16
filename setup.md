# 🛠️ SETUP GUIDE — Vision AI Pipeline

This guide explains how to set up the project from scratch, what every folder does, and how the full pipeline flows from start to finish.

---

## ✅ STEP 1 — First Time Setup

### 1.1 — Create a Python Virtual Environment
A virtual environment keeps all the project packages separate from the rest of your computer.

```bash
python3 -m venv env
source env/bin/activate
```

> You will see `(env)` appear before your terminal line. This means the environment is active.

### 1.2 — Install All Required Packages

```bash
pip3 install -r requirements.txt
```

This installs everything the script needs:
| Package | What it does |
|---|---|
| `opencv-python` | Reads camera videos and saves images |
| `ultralytics` | Runs YOLO AI to find people in images |
| `numpy` | Fast number processing |
| `torch` | Deep Learning engine (PyTorch) |
| `torchvision` | Provides the ResNet18 AI model |
| `Pillow` | Opens and edits images |

---

## 📁 STEP 2 — Folder & File Structure

Below is the **complete project structure** and what every single file and folder does.

```
Automation-modeltraining/
│
├── main.py                ← The main script. Run this to start everything.
├── cameras.json           ← The list of all your cameras and their RTSP links.
├── requirements.txt       ← List of Python packages to install.
├── README.md              ← Overview of the project.
├── setup.md               ← This file. Full setup guide.
│
├── yolov8s.pt             ← The YOLO AI model file (auto-downloaded on first run).
│
├── env/                   ← Virtual environment. Created by you. DO NOT edit manually.
│
├── reference_data/        ← (YOU CREATE THIS) Sample clothing photos for matching.
│   ├── sec1/              ← Put photos of Section 1 staff here
│   ├── sec2/              ← Put photos of Section 2 staff here
│   ├── sec3/
│   ├── sec4/
│   ├── sec5/
│   ├── sec6/
│   ├── sec7/
│   ├── sec8/
│   ├── sec9/
│   └── customers/         ← Put photos of regular customers here
│
├── dataset/               ← (AUTO CREATED) Raw output from cameras while running.
│   └── 2026-06-16/        ← Date folder (created automatically)
│       └── <store-name>/  ← One folder per store
│           ├── images/    ← Good quality camera screenshots saved here
│           ├── blur/      ← Bad/blurry frames that were rejected go here
│           ├── crops/     ← Cropped pictures of each person, sorted by section
│           │   ├── sec2/
│           │   ├── sec3/
│           │   └── customers/
│           └── annotations/
│               ├── images/  ← Images with boxes and labels drawn on them
│               └── txt/     ← YOLO label files (one .txt per image)
│
└── training_dataset/      ← (AUTO CREATED) Final packaged dataset for AI training.
    ├── data.yaml          ← Config file for YOLO training (auto-generated)
    ├── images/
    │   ├── train/         ← 70% of images go here for training
    │   ├── val/           ← 20% of images go here for checking accuracy
    │   └── test/          ← 10% of images go here for final testing
    └── labels/
        ├── train/         ← YOLO label .txt files matching the train images
        ├── val/           ← YOLO label .txt files matching the val images
        └── test/          ← YOLO label .txt files matching the test images
```

---

## 📷 STEP 3 — Setting Up `cameras.json`

This file tells the script which cameras to connect to. Open it and edit it to add your own cameras.

**Format:**
```json
[
    {
        "site_name": "your-store-name",
        "cameras": [
            {
                "camera_id": "CAM-01",
                "rtsp_url": "rtsp://username:password@ip-address:port/stream-path"
            }
        ]
    }
]
```

> ⚠️ You can add multiple stores and multiple cameras per store. Each camera runs at the same time automatically.

---

## 🖼️ STEP 4 — Setting Up `reference_data/`

This is the most important manual step. You need to add sample clothing photos for the AI to learn what each section looks like.

**Rules:**
- Create one folder for each section inside `reference_data/`
- Put **5–10 clear photos** of the uniform for each section (minimum 3)
- Put **10–15 random casual clothing photos** in `customers/`
- Photos should clearly show the uniform — chest to waist area works best
- File formats supported: `.jpg`, `.png`, `.jpeg`

> 📖 See [reference_data_guide.md](./reference_data_guide.md) for full guidelines on what photos to take, how many, and tips for best accuracy.

**Example:**
```
reference_data/
├── sec2/
│   ├── staff_photo_1.jpg
│   ├── staff_photo_2.jpg
│   └── staff_photo_3.jpg
├── customers/
│   ├── customer_1.jpg
│   └── customer_2.jpg
```

> ℹ️ The AI (ResNet18) will automatically read all these photos at startup and learn what each section's clothing looks like.

---

## 🚀 STEP 5 — Running the Script

Once everything is set up, run:

```bash
python3 main.py
```

### What happens when you run it:

```
Step 1 → Script reads cameras.json and loads all camera links
Step 2 → It connects to all cameras at the same time (in parallel)
Step 3 → It takes a photo every 2 seconds from each camera
Step 4 → Quality Check → Blurry or dark images go to the blur/ folder
Step 5 → YOLO AI looks for people in the good images
Step 6 → For each person found, it cuts them out and checks their clothing
Step 7 → ResNet AI compares their clothing to reference_data/ photos
Step 8 → Person is labeled with the correct section name
Step 9 → Annotated image is saved + YOLO .txt label file is saved
Step 10 → After all cameras finish, the training_dataset/ is auto-built
```

### Console Output Explained:
```
  📷  CAMERA CONNECTED  |  GF-37-CAM-01  — Starting capture...
  ✅  CLEAR  |  GF-37-CAM-01  [ 1/10]  |  Sharpness: 9800.1  Brightness: 94.2  Detail: 55.1
  🚫  BLUR   |  GF-37-CAM-01  [ 2/10]  |  Sharpness:   45.2  Brightness:128.0  Detail:  0.4
  ❌  CAMERA OFFLINE  |  GF-35-CAM-05  — Cannot connect.
  🏁  ALL CAMERAS FINISHED — MAX_IMAGES reached for every camera.
```

**What the scores mean:**
- `Sharpness` → How sharp the image is (below 60 = rejected as blurry)
- `Brightness` → Light level (below 40 = too dark, above 220 = too bright = rejected)
- `Detail` → How much detail is visible (below 5 = rejected as blank/empty frame)

---

## ⚙️ STEP 6 — Configuration Settings (in `main.py`)

You can change these settings at the top of `main.py` under `# CONFIGURATION`:

| Setting | Default | What it changes |
|---|---|---|
| `MAX_IMAGES` | `10` | How many good photos to collect per camera |
| `FRAME_INTERVAL` | `2` | Seconds to wait between each photo |
| `BLUR_THRESHOLD` | `150` | Sharpness cutoff — lower = stricter |

---

## 📦 STEP 7 — Using the Training Dataset

After the script finishes, your `training_dataset/` folder is ready. Open the `data.yaml` file to see the class names and paths.

To train a new YOLO model with your data, run:
```bash
yolo train model=yolov8s.pt data=training_dataset/data.yaml epochs=50 imgsz=640
```

---

## 🔁 Running Again

Every time you run `python3 main.py`:
- New images are **added** to the existing `dataset/` folder (organized by date)
- The `training_dataset/` is **rebuilt fresh** from ALL collected data
- The more you run it, the larger and better your training dataset grows!

---

## ❓ Common Issues

| Problem | Solution |
|---|---|
| `❌ CAMERA OFFLINE` | Camera is offline or RTSP link is wrong. Check your `cameras.json`. |
| All images are `🚫 BLUR` | Camera is not focused or stream is initializing. Wait and try again. |
| `0 Labeled images` | No people visible in camera. Check angle or add reference images. |
| `⚠️ No annotated images found` | Cameras ran but detected no people. Check camera angle or reference_data. |
| `ModuleNotFoundError` | Run `pip3 install -r requirements.txt` inside the virtual environment. |
