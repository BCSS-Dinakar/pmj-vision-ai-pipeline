"""
VISION AI PIPELINE SCRIPT
=========================
Automated tool that connects to RTSP security cameras and builds a YOLO dataset.

Steps:
  1. CAMERAS     — Reads camera links from 'cameras.json'
  2. CAPTURE     — Takes screenshots from each camera stream
  3. QUALITY     — Discards blurry, dark, or overexposed frames
  4. DETECTION   — Uses YOLOv8-Seg to detect and segment people
  5. MATCHING    — Compares each person's clothing against reference images
                   using ResNet18 embeddings (shape) + HSV color histograms (color)
  6. ANNOTATION  — Saves YOLO polygon labels + section-labeled annotated images
  7. CROPPING    — Saves transparent PNG cutouts of each person per section
  8. DATASET     — Splits all labeled images into train / val / test folders

HOW TO RUN:
    source env/bin/activate
    python3 main.py

OUTPUTS:
    dataset/           → Daily raw output (images, blur rejects, annotations, crops)
    training_dataset/  → Final YOLO-ready dataset (70% train, 20% val, 10% test)

REQUIRED:
    cameras.json       → RTSP camera links per store
    reference_data/    → Folders (sec1, sec2, … customers) with sample clothing PNGs
    yolov8s-seg.pt     → YOLOv8 segmentation model weights
"""

# ==========================================
# IMPORTS
# ==========================================
import os
import cv2
import json
import time
import pickle
import random
import shutil
import multiprocessing
from datetime import datetime

import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

# Path to cameras config file
JSON_FILE = "cameras.json"

# Seconds to wait between each captured frame per camera
FRAME_INTERVAL = 2

# Laplacian variance below this = image is too blurry to use
BLUR_THRESHOLD = 60

# Stop collecting after this many frames per camera
MAX_IMAGES = 20

# Maps each section label to its YOLO class ID
CLASS_MAPPING = {
    "sec1": 0,
    "sec1-sec2-sec3": 1,
    "sec2": 2,
    "sec3": 3,
    "sec4": 4,
    "sec5": 5,
    "sec6": 6,
    "sec7": 7,
    "customers": 8,
    "sec8": 9,
    "sec10": 10
}


# ==========================================
# UTILITIES
# ==========================================

def create_folder(path):
    """Creates a folder (and any parent folders) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def is_good_frame(image):
    """
    Checks whether a camera frame passes quality checks.

    Rejects frames that are:
      - Blurry       (Laplacian variance < BLUR_THRESHOLD)
      - Too dark     (mean brightness < 40)
      - Too bright   (mean brightness > 220)
      - Low detail   (edge density < 5 — likely a blank/black screen)

    Returns:
      score_dict  — measured values for logging
      is_good     — True if frame passes all checks
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (400, 300))

    blur_score   = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness   = gray.mean()
    edges        = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()

    is_bad = (
        blur_score   < BLUR_THRESHOLD  or
        brightness   < 40  or
        brightness   > 220 or
        edge_density < 5
    )

    score = {
        "blur_score":   blur_score,
        "brightness":   brightness,
        "edge_density": edge_density
    }

    return score, not is_bad


# ==========================================
# CLOTHING MATCHER
# Identifies which store section a person belongs to by comparing
# their segmented clothing against reference images.
#
# Matching uses two signals combined:
#   60% — ResNet18 embedding (texture / clothing shape)
#   40% — HSV color histogram (dominant clothing color)
#
# Reference embeddings are cached to 'reference_cache.pkl' on first
# run and reloaded instantly on all subsequent runs.
#
# The alpha mask from RGBA crops ensures only the person's actual
# pixels contribute to both signals — background is fully ignored.
# ==========================================

class ReferenceMatcher:

    def __init__(self, ref_dir="reference_data", threshold=0.75):
        self.ref_dir   = ref_dir
        self.threshold = threshold

        # Use the fastest available hardware
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        # Load ResNet18 pretrained on ImageNet; strip the classifier head
        # so the output is a 512-dim feature vector (embedding)
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            self.model      = resnet18(weights=weights)
            self.preprocess = weights.transforms()
        except ImportError:
            from torchvision.models import resnet18
            self.model = resnet18(pretrained=True)
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.model.fc = torch.nn.Identity()
        self.model    = self.model.to(self.device)
        self.model.eval()

        # Stores (embedding, histogram, class_id) for every reference image
        self.reference_embeddings = []

        cache_file = "reference_cache.pkl"

        if os.path.exists(cache_file):
            print("Loading reference cache...")
            with open(cache_file, "rb") as f:
                self.reference_embeddings = pickle.load(f)
        else:
            self._load_references()
            with open(cache_file, "wb") as f:
                pickle.dump(self.reference_embeddings, f)

        print("Total reference images:", len(self.reference_embeddings))

    def _compute_color_hist(self, image):
        """
        Computes a normalized 2D HSV histogram (Hue × Saturation).
        If the image has an alpha channel, uses it as a mask so that only
        the person's pixels are counted and the background is ignored.
        """
        if len(image.shape) == 3 and image.shape[2] == 4:
            bgr  = image[:, :, :3]
            mask = image[:, :, 3]
        else:
            bgr  = image
            mask = None

        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _compute_embedding(self, image):
        """
        Converts an image (BGR or RGBA) into an L2-normalized ResNet18 embedding.
        For RGBA images, the transparent background is replaced with white before
        passing through ResNet so the clothing pixels dominate.
        """
        # Flatten alpha channel onto a white background for ResNet input
        if len(image.shape) == 3 and image.shape[2] == 4:
            rgb        = image[:, :, :3]
            alpha      = image[:, :, 3]
            background = np.ones_like(rgb) * 255
            image      = np.where(alpha[:, :, None] > 0, rgb, background)

        # BGR → RGB → PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(image_rgb)

        # Crop the inner 80% to reduce edge noise / background bleed
        width, height = pil_img.size
        if width > 0 and height > 0:
            pil_img = pil_img.crop((
                int(width  * 0.1), int(height * 0.1),
                int(width  * 0.9), int(height * 0.9)
            ))

        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def _load_references(self):
        """
        Scans reference_data/<section>/ and builds (embedding, histogram, class_id)
        tuples for every image found. Called only when no cache file exists.
        Results are stored in self.reference_embeddings and later pickled
        to 'reference_cache.pkl' by __init__ for fast reuse on future runs.

        Uses cv2.IMREAD_UNCHANGED so alpha channels in PNGs are preserved.
        """
        if not os.path.exists(self.ref_dir):
            print("Reference folder not found")
            return

        for folder_name, class_id in CLASS_MAPPING.items():
            folder_path = os.path.join(self.ref_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for filename in os.listdir(folder_path):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                image_path = os.path.join(folder_path, filename)
                # IMREAD_UNCHANGED keeps the alpha channel intact
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue

                emb  = self._compute_embedding(image)
                hist = self._compute_color_hist(image)
                self.reference_embeddings.append((emb, hist, class_id))
                print(f"Loaded reference -> {folder_name}/{filename}")

    def match(self, cropped_img):
        """
        Compares a segmented RGBA person crop against every cached reference fingerprint.
        Score = (ResNet cosine similarity × 0.6) + (HSV color histogram correlation × 0.4).

        Returns:
          (class_id, best_score) — class_id is the winning section if the score exceeds
          self.threshold (default 0.75), otherwise falls back to 'customers'.
        """
        if not self.reference_embeddings:
            return CLASS_MAPPING.get("customers", 8), 0.0

        crop_emb  = self._compute_embedding(cropped_img)
        crop_hist = self._compute_color_hist(cropped_img)
        best_score = -1.0
        best_class = CLASS_MAPPING.get("customers", 8)

        for ref_emb, ref_hist, class_id in self.reference_embeddings:
            resnet_score = torch.sum(crop_emb * ref_emb).item()
            color_score  = cv2.compareHist(crop_hist, ref_hist, cv2.HISTCMP_CORREL)
            score        = (resnet_score * 0.6) + (color_score * 0.4)

            if score > best_score:
                best_score = score
                if score > self.threshold:
                    best_class = class_id

        return best_class, best_score


# ==========================================
# ANNOTATION FUNCTION
# ==========================================

def save_yolo_annotation(results, txt_file, image, matcher,
                         crop_folder=None, segmented_crop_folder=None, base_name=""):
    """
    Processes every detected person in a YOLO result:
      1. Filters out tiny / irrelevant detections
      2. Builds an RGBA segmented crop using the YOLO mask
      3. Matches the crop against reference clothing via ReferenceMatcher
      4. Appends the YOLO polygon label line (or bbox fallback) to the txt file
      5. Draws the section label + bounding box on the annotated image
      6. Saves both a regular crop (JPG) and a transparent crop (PNG)

    Returns (annotated_image, num_valid_detections).
    """
    img_h, img_w  = image.shape[:2]
    lines         = []
    annotated_image = image.copy()
    ID_TO_CLASS   = {v: k for k, v in CLASS_MAPPING.items()}

    for idx, box in enumerate(results[0].boxes):
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_width  = x2 - x1
        box_height = y2 - y1
        box_area   = box_width * box_height
        image_area = img_w * img_h

        # Skip detections that are too small to be reliable
        if box_height < 80:
            continue
        if (box_area / image_area) < 0.01:
            continue

        x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
        x2_c, y2_c = min(img_w, int(x2)), min(img_h, int(y2))

        normal_cropped = image[y1_c:y2_c, x1_c:x2_c]

        # Build RGBA crop using the segmentation mask
        if results[0].masks is not None and len(results[0].masks.data) > idx:
            mask = results[0].masks.data[idx].cpu().numpy()
            # Ensure mask is the size of the full image before cropping
            if mask.shape[0] != img_h or mask.shape[1] != img_w:
                mask = cv2.resize(mask, (img_w, img_h))
            
            mask = (mask > 0.5).astype("uint8")
            
            # Crop the mask to the person's bounding box
            cropped_mask = mask[y1_c:y2_c, x1_c:x2_c]
            
            # Convert ONLY the cropped image to RGBA (much faster than full image)
            rgba = cv2.cvtColor(normal_cropped, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = cropped_mask * 255
            segmented_cropped_rgba = rgba
        else:
            segmented_cropped_rgba = normal_cropped

        # Match section using the transparent RGBA crop
        if segmented_cropped_rgba.size == 0:
            class_id, score = CLASS_MAPPING.get("customers", 8), 0.0
        else:
            class_id, score = matcher.match(segmented_cropped_rgba)

        print(
            "Matched:",
            ID_TO_CLASS[class_id],
            "Score:",
            round(score, 3)
        )

        # Save YOLO polygon label if masks exist, otherwise save bbox label
        if results[0].masks is not None and len(results[0].masks.xyn) > idx:
            polygon  = results[0].masks.xyn[idx]
            poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in polygon])
            lines.append(f"{class_id} {poly_str}")
        else:
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label = ID_TO_CLASS.get(class_id, "unknown")
        cv2.rectangle(annotated_image, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2)
        cv2.putText(annotated_image, label, (x1_c, max(0, y1_c - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save regular JPG crop
        if crop_folder and base_name and normal_cropped.size > 0:
            label_crop_folder = os.path.join(crop_folder, label)
            create_folder(label_crop_folder)
            cv2.imwrite(os.path.join(label_crop_folder, f"{idx}_{base_name}"), normal_cropped)

        # Save transparent PNG segmented crop
        if segmented_crop_folder and base_name and segmented_cropped_rgba.size > 0:
            label_seg_folder = os.path.join(segmented_crop_folder, label)
            create_folder(label_seg_folder)
            out_name = f"{idx}_{base_name}".replace(".jpg", ".png")
            cv2.imwrite(os.path.join(label_seg_folder, out_name), segmented_cropped_rgba)

    # Only write the txt file if at least one valid person was found
    if len(lines) > 0:
        create_folder(os.path.dirname(txt_file))
        with open(txt_file, "w") as f:
            f.write("\n".join(lines))

    return annotated_image, len(lines)


# ==========================================
# CAMERA LOOP
# Runs as a separate process per camera
# ==========================================

def process_camera(site_name, camera_id, rtsp_url, return_dict):
    """
    Runs as a separate subprocess for one camera. Captures frames in a loop
    until MAX_IMAGES frames have been processed (good + bad combined).

    Per frame:
      - Quality check via is_good_frame() — rejects blurry / dark / overexposed frames
      - Bad frames   → saved to blur/ only
      - Good frames  → YOLO detection → section matching → annotation + crop saving
      - Empty frames → no files written (folders are never created until data exists)

    Prints per-person match label and score to the console during processing.
    Updates the shared return_dict with per-camera stats for the final summary.
    """
    matcher = ReferenceMatcher()
    model   = YOLO("yolov8s-seg.pt")

    return_dict[camera_id] = {"site_name": site_name, "clear": 0, "blur": 0, "annotated": 0, "persons": 0}

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"\n  ❌  CAMERA OFFLINE  |  {camera_id}  — Cannot connect. Check RTSP link or network.")
        stats = return_dict[camera_id]
        stats["error"] = "Connection failed (Camera offline or unreachable)"
        return_dict[camera_id] = stats
        return

    print(f"\n  📷  CAMERA CONNECTED  |  {camera_id}  — Starting capture...")
    count = 0

    while count < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print(f"  🔄  RECONNECTING  |  {camera_id}  — Frame read failed. Retrying in 5s...")
            time.sleep(5)
            continue

        date   = datetime.now().strftime("%Y-%m-%d")
        base   = f"dataset/{date}/{site_name}"
        image_folder          = f"{base}/images"
        blur_folder           = f"{base}/blur"
        ann_img_folder        = f"{base}/annotations/images"
        ann_txt_folder        = f"{base}/annotations/txt"
        crop_folder           = f"{base}/crops"
        segmented_crop_folder = f"{base}/segmented_crops"

        score_dict, is_good = is_good_frame(frame)
        name = f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        if not is_good:
            # Bad frame — save to blur/ for review
            create_folder(blur_folder)
            cv2.imwrite(f"{blur_folder}/{name}", frame)
            status = "BLUR"
            stats = return_dict[camera_id]
            stats["blur"] += 1
            return_dict[camera_id] = stats

        else:
            results  = model(frame, classes=[0], conf=0.5, verbose=False, retina_masks=True)
            txt_name = name.replace(".jpg", ".txt")

            annotated, num_boxes = save_yolo_annotation(
                results, f"{ann_txt_folder}/{txt_name}",
                frame, matcher, crop_folder, segmented_crop_folder, name
            )

            if num_boxes > 0:
                # Only create image/annotation folders and save files if people were detected
                create_folder(image_folder)
                create_folder(ann_img_folder)
                cv2.imwrite(f"{image_folder}/{name}", frame)
                cv2.imwrite(f"{ann_img_folder}/{name}", annotated)

                status = "CLEAR"
                stats  = return_dict[camera_id]
                stats["clear"]     += 1
                stats["annotated"] += 1
                stats["persons"]   += num_boxes
                return_dict[camera_id] = stats
            else:
                status = "EMPTY"

        count += 1

        icon = "✅" if status == "CLEAR" else "🚫"
        print(
            f"  {icon}  {status:<5}  |  {camera_id}  [{count:>2}/{MAX_IMAGES}]"
            f"  |  Sharpness:{score_dict['blur_score']:>8.1f}"
            f"  Brightness:{score_dict['brightness']:>5.1f}"
            f"  Detail:{score_dict['edge_density']:>5.1f}"
        )

        time.sleep(FRAME_INTERVAL)

    cap.release()


# ==========================================
# DATASET BUILDER
# Packages annotated images into the YOLO training_dataset/ structure
# ==========================================

def create_training_dataset(source_dir="dataset", dest_dir="training_dataset", split_ratios=(0.7, 0.2, 0.1)):
    """
    Scans dataset/ for all images that have a non-empty YOLO annotation file,
    then copies them into training_dataset/ split into train / val / test.

    Split ratios (default):
      70% → train    (model learning)
      20% → val      (accuracy monitoring during training)
      10% → test     (final evaluation after training)

    Also generates data.yaml required by YOLO training.
    """
    print(f"\n{'='*55}")
    print(f"  📦  Building Training Dataset in '{dest_dir}'...")
    print(f"{'='*55}")

    for split in ['train', 'val', 'test']:
        create_folder(f"{dest_dir}/{split}/images")
        create_folder(f"{dest_dir}/{split}/labels")

    if not os.path.exists(source_dir):
        print(f"  ⚠️  ERROR: Source folder '{source_dir}' not found.")
        return

    all_images = []

    for date_folder in os.listdir(source_dir):
        date_path = os.path.join(source_dir, date_folder)
        if not os.path.isdir(date_path):
            continue

        for site_folder in os.listdir(date_path):
            site_path = os.path.join(date_path, site_folder)
            if not os.path.isdir(site_path):
                continue

            img_dir = os.path.join(site_path, "images")
            txt_dir = os.path.join(site_path, "annotations", "txt")

            if not os.path.exists(img_dir) or not os.path.exists(txt_dir):
                continue

            for img_name in os.listdir(img_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                txt_name = img_name.rsplit('.', 1)[0] + ".txt"
                img_path = os.path.join(img_dir, img_name)
                txt_path = os.path.join(txt_dir, txt_name)

                # Only include images that have a non-empty label file
                if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                    all_images.append((img_path, txt_path, img_name, txt_name))

    if not all_images:
        print("  ⚠️  WARNING: No annotated images found.")
        return

    random.seed(42)
    random.shuffle(all_images)

    total     = len(all_images)
    train_end = int(total * split_ratios[0])
    val_end   = train_end + int(total * split_ratios[1])

    def copy_split(data, split_name):
        for img_path, txt_path, img_name, txt_name in data:
            shutil.copy(img_path, os.path.join(dest_dir, split_name, "images", img_name))
            shutil.copy(txt_path, os.path.join(dest_dir, split_name, "labels", txt_name))

    copy_split(all_images[:train_end],        'train')
    copy_split(all_images[train_end:val_end], 'val')
    copy_split(all_images[val_end:],          'test')

    names_yaml   = "\n".join([f"  {class_id}: {name}" for name, class_id in CLASS_MAPPING.items()])
    yaml_content = f"train: train/images\nval: val/images\ntest: test/images\n\nnames:\n{names_yaml}\n"

    with open(os.path.join(dest_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"  ✅  Dataset ready!")
    print(f"  🏋️  Train : {train_end} images")
    print(f"  🔍  Val   : {val_end - train_end} images")
    print(f"  🧪  Test  : {total - val_end} images")
    print(f"  📄  data.yaml generated at: {dest_dir}/data.yaml")


# ==========================================
# ENTRY POINT
# ==========================================

def main():
    with open(JSON_FILE) as f:
        data = json.load(f)

    manager     = multiprocessing.Manager()
    return_dict = manager.dict()
    processes   = []

    # Launch one process per camera so all cameras run in parallel
    for site in data:
        for cam in site["cameras"]:
            p = multiprocessing.Process(
                target=process_camera,
                args=(site["site_name"], cam["camera_id"], cam["rtsp_url"], return_dict)
            )
            p.start()
            processes.append(p)

    try:
        while any(p.is_alive() for p in processes):
            time.sleep(1)
        print("\n" + "="*55)
        print("  🏁  ALL CAMERAS FINISHED — MAX_IMAGES reached for every camera.")
        print("="*55)
    except KeyboardInterrupt:
        print("\n  🛑  STOPPED — Cameras terminated by user (Ctrl+C).")
        for p in processes:
            p.terminate()

    # Print per-camera summary
    print("\n" + "═"*55)
    print("   📊  RUN SUMMARY")
    print("═"*55)

    total_clear = total_blur = total_anno = total_persons = 0

    for cam_id, stats in return_dict.items():
        print(f"\n  🏪  {stats['site_name']}  |  📷 {cam_id}")
        if "error" in stats:
            print(f"      ❌  Status   : ERROR — {stats['error']}")
        print(f"      ✅  Clear    : {stats['clear']} images")
        print(f"      🚫  Rejected : {stats['blur']} images (blur/dark/overexposed)")
        print(f"      🏷️  Labeled  : {stats['annotated']} images with annotations")
        print(f"      👤  Persons  : {stats['persons']} people detected")

        total_clear   += stats.get('clear', 0)
        total_blur    += stats.get('blur', 0)
        total_anno    += stats.get('annotated', 0)
        total_persons += stats.get('persons', 0)

    print("\n" + "─"*55)
    print(f"  ✅  Total Clear Images   : {total_clear}")
    print(f"  🚫  Total Rejected       : {total_blur}")
    print(f"  🏷️  Total Labeled        : {total_anno}")
    print(f"  👤  Total Persons Found  : {total_persons}")
    print("═"*55)

    create_training_dataset()


if __name__ == "__main__":
    main()