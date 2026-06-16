# Vision AI Pipeline: Automated Dataset Generator

This repository contains an automated pipeline designed to connect to multiple RTSP camera streams, capture high-quality frames, detect people, classify their uniforms/sections, and automatically generate a perfectly formatted YOLO training dataset.

## Features

- **Multi-Camera RTSP Capture**: Simultaneously connects to and processes streams from multiple cameras listed in `cameras.json` using Python's `multiprocessing`.
- **Advanced Quality Checks**: Automatically filters out bad frames by mathematically analyzing:
  - Sharpness (Laplacian Variance)
  - Brightness (Overexposure & Underexposure thresholds)
  - Noise/Low Detail (Canny Edge Density)
- **YOLO Person Detection**: Uses Ultralytics YOLOv8 to detect people in clear frames.
- **Deep Learning Reference Matching**: Employs a pre-trained PyTorch `ResNet18` model to extract visual features from detected people (center-cropped to ignore background noise) and matches them against `reference_data` using Cosine Similarity.
- **Automated Dataset Structuring**: Automatically filters out empty frames (where no person is detected) and splits the remaining annotated images into an exact 70% Train, 20% Val, and 10% Test split inside a YOLO-compatible `training_dataset/` directory. It also dynamically generates a precise `data.yaml` configuration using dictionary-mapping to safely skip unused class IDs.

## Setup & Installation

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

## Requirements

Ensure the following directories and files exist before running the script:

- `cameras.json`: Contains the JSON array mapping sites to their respective camera RTSP URLs.
- `reference_data/`: Contains the reference images for each clothing section used by the ResNet matching engine. Structure it like so:
  - `reference_data/sec1/`
  - `reference_data/sec2/`
  - `reference_data/customers/`

*(Note: Data output folders like `dataset/` and `training_dataset/` are automatically created and git-ignored to keep the repository clean).*

## Usage

Simply run the main script. It will connect to all defined cameras, process frames, discard bad images to a `blur/` folder, generate YOLO `.txt` labels, and compile the final split dataset.

```bash
python3 main.py
```

## Output Artifacts

After execution, the following folders are generated:
- `dataset/`: Contains raw date/site categorized images, blur rejects, raw annotation text files, and section-wise cropped person images.
- `training_dataset/`: The final, filtered, and securely split dataset ready to be directly plugged into a YOLO training script. This folder also contains the auto-generated `data.yaml` mapping your exact section classes.
