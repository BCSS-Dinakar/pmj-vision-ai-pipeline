"""
VISION AI PIPELINE SCRIPT
=========================
This script is an automated tool that connects to security cameras and builds a dataset for AI training.

Here is exactly what it does, step by step:
1. READ CAMERAS  : It reads a list of camera links from 'cameras.json'.
2. TAKE PHOTOS   : It connects to these cameras and takes screenshots (frames) every 2 seconds.
3. QUALITY CHECK : It automatically throws away any blurry, too dark, or too bright images.
4. FIND PEOPLE   : It uses a smart AI called 'YOLO' to draw boxes around any people it sees.
5. MATCH SECTION : For each person, it looks at their clothes and compares them to your 'reference_data' photos to figure out which section they belong to (e.g. 'sec1', 'sec2', 'customers').
6. SAVE LABELS   : It draws the section name on the image and saves a matching YOLO .txt label file.
7. BUILD DATASET : After all cameras finish, it collects all good labeled images and splits them into train/val/test folders, ready for YOLO model training!

HOW TO RUN:
    source env/bin/activate
    python3 main.py

FOLDERS CREATED:
    dataset/           → Raw daily output (images, blur rejects, annotations, crops)
    training_dataset/  → Final clean dataset for YOLO training (70% train, 20% val, 10% test)

FILES YOU NEED:
    cameras.json       → List of all your camera RTSP links
    reference_data/    → Folders (sec1, sec2, ... customers) with sample clothing photos
"""

# ==========================================
# IMPORTS
# Standard Python tools for file/folder handling, timing, and randomizing
# ==========================================
import os
import cv2          # OpenCV — reads camera videos and saves images
import json         # Reads cameras.json file
import time         # Used to wait between camera frames
import random       # Used to randomly shuffle images before splitting into train/val/test
import shutil       # Used to copy files into training_dataset folder
import multiprocessing  # Runs each camera in parallel (at the same time)
from datetime import datetime  # Gets today's date for folder naming

import torch                             # PyTorch — runs Deep Learning models
import numpy as np                       # Fast number/array processing
import torchvision.transforms as transforms  # Image preprocessing for ResNet
from PIL import Image                    # Opens images for ResNet processing
from ultralytics import YOLO             # YOLO — person detection AI


# ==========================================
# CONFIGURATION
# Change these values to control how the script behaves
# ==========================================

# The file that contains all your camera links
JSON_FILE = "cameras.json"

# How many seconds to wait between each photo from a camera
FRAME_INTERVAL = 2

# Sharpness limit — images below this are considered "blurry" and discarded
BLUR_THRESHOLD = 150

# How many good photos to collect per camera before stopping
MAX_IMAGES = 10

# Maps each section name to a unique number (class ID) for YOLO training
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
# Small helper functions used throughout the script
# ==========================================

def create_folder(path):
    """Creates a folder at the given path. Does nothing if the folder already exists."""
    os.makedirs(path, exist_ok=True)


def is_good_frame(image):
    """
    Checks if a camera frame (photo) is good enough to use.

    It checks 4 things:
      - Is it blurry?         (blur_score below 60 = bad)
      - Is it too dark?       (brightness below 40 = bad)
      - Is it too bright?     (brightness above 220 = bad)
      - Is it low detail?     (edge_density below 5 = bad, usually a black screen)

    Returns:
      score  → dictionary with the measured values (for printing)
      True   → if the image is good
      False  → if the image is bad and should be discarded
    """
    # Convert to grayscale and resize for fast processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (400, 300))

    # Measure sharpness: high number = sharp, low number = blurry
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Measure brightness: 0 = completely black, 255 = completely white
    brightness = gray.mean()

    # Measure how much detail/edges are visible (low = possibly a black/blank frame)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()

    # Apply the 4 rules to decide if the frame is bad
    is_blur       = blur_score < 60
    is_dark       = brightness < 40
    is_bright     = brightness > 220
    is_low_detail = edge_density < 5

    is_bad = is_blur or is_dark or is_bright or is_low_detail

    score = {
        "blur_score":   blur_score,
        "brightness":   brightness,
        "edge_density": edge_density
    }

    return score, not is_bad   # Returns (score_dict, is_good)


# ==========================================
# REFERENCE MATCHER (DEEP LEARNING)
# This AI reads clothes photos and tells which section a person belongs to
# ==========================================

class ReferenceMatcher:
    """
    This class compares a cropped photo of a person against the sample clothing photos
    in your 'reference_data/' folder to figure out which section they work in.

    HOW IT WORKS:
      1. At startup, it reads all photos from 'reference_data/sec1/', 'reference_data/sec2/', etc.
      2. It passes each photo through ResNet18 (a Deep Learning model) to get a unique
         "fingerprint" (called an embedding) for that photo.
      3. When a person is detected in a camera frame, their cropped image is also
         turned into a fingerprint.
      4. It compares the person's fingerprint to all the reference fingerprints using
         a math formula called Cosine Similarity.
      5. Whichever reference image matches best (above the 0.75 threshold), that
         section is assigned to the person.
      6. If no match is found, the person is labeled as 'customers'.
    """

    def __init__(self, ref_dir="reference_data", threshold=0.75):
        # The folder where your reference clothing photos are stored
        self.ref_dir = ref_dir

        # How similar the clothing must be to count as a match (0 to 1, higher = stricter)
        self.threshold = threshold

        # Automatically use the fastest available hardware: NVIDIA GPU > Apple GPU > CPU
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        # Load the ResNet18 model (pretrained on millions of images for good general vision)
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
            self.preprocess = weights.transforms()   # Standard image preparation steps
        except ImportError:
            # Fallback for older versions of torchvision
            from torchvision.models import resnet18
            self.model = resnet18(pretrained=True)
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Remove the final classification layer so we get raw 512-number feature vectors
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()   # Set model to inference mode (no training)

        # This list will store the pre-computed fingerprints of all reference photos
        self.reference_embeddings = []

        # Load and process all reference photos at startup
        self._load_references()

    def _compute_embedding(self, image):
        """
        Converts a photo into a 512-number fingerprint using ResNet18.
        This fingerprint represents the visual "style" of the clothing.
        """
        # Convert from OpenCV format (BGR) to standard RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Crop the inner 80% of the image to ignore edges/background noise
        # This focuses the AI on the clothing in the center
        width, height = pil_img.size
        if width > 0 and height > 0:
            left, top   = width * 0.1, height * 0.1
            right, bottom = width * 0.9, height * 0.9
            pil_img = pil_img.crop((left, top, right, bottom))

        # Prepare the image and send it through ResNet18
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():   # Don't calculate gradients (faster, saves memory)
            embedding = self.model(input_tensor)

        # Normalize the fingerprint so similarity scores stay between -1 and 1
        return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def _load_references(self):
        """
        Reads all photos from 'reference_data/' at startup and converts each
        one into a fingerprint. These are stored in memory for fast matching later.
        """
        if not os.path.exists(self.ref_dir):
            return   # No reference folder found, skip

        for folder_name, class_id in CLASS_MAPPING.items():
            folder_path = os.path.join(self.ref_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue   # Skip if this section folder doesn't exist

            for filename in os.listdir(folder_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue   # Skip non-image files

                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Compute and store the fingerprint + its class ID
                    emb = self._compute_embedding(image)
                    self.reference_embeddings.append((emb, class_id))

    def match(self, cropped_img):
        """
        Compares a cropped person photo against all stored reference fingerprints.
        Returns the class ID (section number) of the best match.
        If nothing matches above the threshold, returns the 'customers' class ID.
        """
        # If no reference photos were loaded, everyone defaults to 'customers'
        if not self.reference_embeddings:
            return CLASS_MAPPING.get("customers", 8)

        crop_emb = self._compute_embedding(cropped_img)
        best_score = -1.0
        best_class = CLASS_MAPPING.get("customers", 8)   # Default = customers

        # Compare the person's fingerprint against every reference fingerprint
        for ref_emb, class_id in self.reference_embeddings:
            # Cosine similarity: 1.0 = identical, 0.0 = completely different
            score = torch.sum(crop_emb * ref_emb).item()
            if score > best_score:
                best_score = score
                if score > self.threshold:   # Only accept if above confidence threshold
                    best_class = class_id

        return best_class


# ==========================================
# ANNOTATION FUNCTION
# Saves bounding box labels and draws section names on the image
# ==========================================

def save_yolo_annotation(results, txt_file, image, matcher, crop_folder=None, base_name=""):
    """
    For every person detected by YOLO in the image:
      1. Crops the person out of the photo
      2. Sends the crop to the ReferenceMatcher to get their section name
      3. Writes the YOLO format bounding box + class ID to a .txt file
      4. Draws a green box and the section name label on the image
      5. Saves the cropped person photo into the correct section subfolder

    Returns the annotated image (with boxes drawn on it).
    """
    img_h, img_w = image.shape[:2]
    lines = []   # Will hold the YOLO label lines for this image
    annotated_image = image.copy()   # Work on a copy so the original stays clean

    # Reverse the CLASS_MAPPING so we can look up section name by class ID
    ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

    for idx, box in enumerate(results[0].boxes):
        # YOLO detects many types of objects; class 0 = person
        if int(box.cls[0]) == 0:
            # Get the pixel coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            image_area = img_w * img_h

            # =========================
            # REMOVE SMALL DETECTIONS
            # =========================

            # ignore very small boxes
            if box_height < 80:
                continue

            # ignore tiny area objects
            if (box_area / image_area) < 0.01:
                continue

            # Clamp the coordinates so they don't go outside the image edges
            x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
            x2_c, y2_c = min(img_w, int(x2)), min(img_h, int(y2))

            # Cut out just the person from the full image
            cropped = image[y1_c:y2_c, x1_c:x2_c]

            # If crop is valid, match the clothing. Otherwise, default to 'customers'
            class_id = CLASS_MAPPING.get("customers", 8) if cropped.size == 0 else matcher.match(cropped)

            # Convert pixel coordinates to YOLO format (center x, center y, width, height)
            # YOLO uses values between 0.0 and 1.0 relative to image size
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            # Draw the green bounding box and section label on the image
            label = ID_TO_CLASS.get(class_id, "unknown")
            cv2.rectangle(annotated_image, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1_c, max(0, y1_c - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the cropped person photo into a subfolder named after their section
            if crop_folder and base_name and cropped.size > 0:
                label_crop_folder = os.path.join(crop_folder, label)
                create_folder(label_crop_folder)
                cv2.imwrite(os.path.join(label_crop_folder, f"{idx}_{base_name}"), cropped)

    # Write all the label lines to the YOLO .txt file
    with open(txt_file, "w") as f:
        f.write("\n".join(lines))

    return annotated_image


# ==========================================
# CAMERA LOOP
# Runs once per camera, captures frames and processes them
# ==========================================

def process_camera(site_name, camera_id, rtsp_url, return_dict):
    """
    This function runs as a completely separate process for each camera.
    It keeps taking photos until it has collected MAX_IMAGES good (non-blurry) images.

    For each photo it:
      - Checks image quality
      - If BAD  → saves to blur/ folder
      - If GOOD → runs YOLO detection → saves annotated image + YOLO label file
    """
    # Each camera process loads its own copy of the models
    # This is important for safe parallel processing on macOS
    matcher = ReferenceMatcher()   # Loads reference clothing photos and builds fingerprints
    model   = YOLO("yolov8s.pt")   # Loads the YOLO person detection model (Small = more accurate)

    # Initialize this camera's stats counter in the shared results dictionary
    return_dict[camera_id] = {"site_name": site_name, "clear": 0, "blur": 0, "annotated": 0, "persons": 0}

    # Open the RTSP camera stream
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Keep buffer small to always get the latest frame

    # If camera is offline or the link is wrong, log the error and stop
    if not cap.isOpened():
        print(f"\n  ❌  CAMERA OFFLINE  |  {camera_id}  — Cannot connect. Check RTSP link or network.")
        stats = return_dict[camera_id]
        stats["error"] = "Connection failed (Camera offline or unreachable)"
        return_dict[camera_id] = stats
        return

    print(f"\n  📷  CAMERA CONNECTED  |  {camera_id}  — Starting capture...")
    count = 0   # Counts how many frames have been processed (both clear and blur)

    # Keep looping until we've processed MAX_IMAGES frames
    while count < MAX_IMAGES:
        ret, frame = cap.read()   # Read the next frame from the camera
        if not ret:
            # Frame failed to read — camera may have dropped. Wait and retry.
            print(f"  🔄  RECONNECTING  |  {camera_id}  — Frame read failed. Retrying in 5s...")
            time.sleep(5)
            continue

        # Build the folder paths for today's date and this store
        date = datetime.now().strftime("%Y-%m-%d")
        base = f"dataset/{date}/{site_name}"
        image_folder   = f"{base}/images"          # Good frames saved here
        blur_folder    = f"{base}/blur"             # Bad/blurry frames saved here
        ann_img_folder = f"{base}/annotations/images"  # Annotated (boxed) images
        ann_txt_folder = f"{base}/annotations/txt"     # YOLO .txt label files
        crop_folder    = f"{base}/crops"               # Cropped person photos by section

        # Make sure all these folders exist
        for folder in [image_folder, blur_folder, ann_img_folder, ann_txt_folder, crop_folder]:
            create_folder(folder)

        # Check if the frame is good enough to use
        score_dict, is_good = is_good_frame(frame)
        name = f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"   # Unique filename

        if not is_good:
            # Frame is blurry / too dark / too bright → save to blur folder and skip
            cv2.imwrite(f"{blur_folder}/{name}", frame)
            status = "BLUR"
            stats = return_dict[camera_id]
            stats["blur"] += 1
            return_dict[camera_id] = stats

        else:
            # Frame is clear → save it, detect people, annotate, and crop
            image_path = f"{image_folder}/{name}"
            cv2.imwrite(image_path, frame)

            # Run YOLO to find all people in the image
            # classes=[0] → only look for class 0 (person)
            # conf=0.25   → detect even people that are partially visible or faraway
            results = model(image_path, classes=[0], conf=0.5, verbose=False)
            txt_name = name.replace(".jpg", ".txt")

            # Save labels, draw boxes, and save section crops
            annotated = save_yolo_annotation(results, f"{ann_txt_folder}/{txt_name}", frame, matcher, crop_folder, name)
            cv2.imwrite(f"{ann_img_folder}/{name}", annotated)
            status = "CLEAR"

            # Count how many people were found and update the stats
            person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
            stats = return_dict[camera_id]
            stats["clear"] += 1
            if person_count > 0:
                stats["annotated"] += 1       # Image had at least 1 person → will go to training dataset
                stats["persons"] += person_count
            return_dict[camera_id] = stats

        count += 1

        # Print a one-line status update to the terminal for this frame
        icon = "✅" if status == "CLEAR" else "🚫"
        print(f"  {icon}  {status:<5}  |  {camera_id}  [{count:>2}/{MAX_IMAGES}]  |  Sharpness:{score_dict['blur_score']:>8.1f}  Brightness:{score_dict['brightness']:>5.1f}  Detail:{score_dict['edge_density']:>5.1f}")

        # Wait before taking the next photo
        time.sleep(FRAME_INTERVAL)

    cap.release()   # Close the camera connection cleanly


# ==========================================
# DATASET BUILDER
# Runs after all cameras finish — packages everything into training_dataset/
# ==========================================

def create_training_dataset(source_dir="dataset", dest_dir="training_dataset", split_ratios=(0.7, 0.2, 0.1)):
    """
    Scans the entire 'dataset/' folder, finds all images that have valid YOLO
    label files (meaning people were found and annotated), and copies them into
    'training_dataset/' split into train / val / test folders.

    Split ratio:
      70% → training_dataset/images/train/ (used to train the model)
      20% → training_dataset/images/val/   (used to check accuracy while training)
      10% → training_dataset/images/test/  (used for final evaluation after training)

    Also automatically generates 'training_dataset/data.yaml' which YOLO needs.
    """
    print(f"\n{'='*55}")
    print(f"  📦  Building Training Dataset in '{dest_dir}'...")
    print(f"{'='*55}")

    # Create the required train/val/test folders inside training_dataset/
    for split in ['train', 'val', 'test']:
        create_folder(f"{dest_dir}/images/{split}")
        create_folder(f"{dest_dir}/labels/{split}")

    # Check that the source dataset folder exists
    if not os.path.exists(source_dir):
        print(f"  ⚠️  ERROR: Source folder '{source_dir}' not found. Run the camera script first.")
        return

    all_images = []   # Will hold tuples of (image_path, label_path, img_name, txt_name)

    # Walk through every date folder and every store folder inside dataset/
    for date_folder in os.listdir(source_dir):
        date_path = os.path.join(source_dir, date_folder)
        if not os.path.isdir(date_path): continue

        for site_folder in os.listdir(date_path):
            site_path = os.path.join(date_path, site_folder)
            if not os.path.isdir(site_path): continue

            img_dir = os.path.join(site_path, "images")
            txt_dir = os.path.join(site_path, "annotations", "txt")

            # Skip this store if either the images or labels folder is missing
            if not os.path.exists(img_dir) or not os.path.exists(txt_dir): continue

            for img_name in os.listdir(img_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

                txt_name = img_name.rsplit('.', 1)[0] + ".txt"
                img_path = os.path.join(img_dir, img_name)
                txt_path = os.path.join(txt_dir, txt_name)

                # Only include images that have a non-empty annotation file
                # (empty file = YOLO found no people in that image)
                if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                    all_images.append((img_path, txt_path, img_name, txt_name))

    if not all_images:
        print("  ⚠️  WARNING: No annotated images found. Make sure cameras captured people.")
        return

    # Shuffle images randomly so the split is not biased by date or camera
    random.seed(42)   # Fixed seed = same shuffle every time (reproducible)
    random.shuffle(all_images)

    # Calculate where each split ends
    total     = len(all_images)
    train_end = int(total * split_ratios[0])          # 70% for train
    val_end   = train_end + int(total * split_ratios[1])  # 20% for val

    # Copy image + label into the correct split folder
    def copy_split(data, split_name):
        for img_path, txt_path, img_name, txt_name in data:
            shutil.copy(img_path, os.path.join(dest_dir, "images", split_name, img_name))
            shutil.copy(txt_path, os.path.join(dest_dir, "labels", split_name, txt_name))

    copy_split(all_images[:train_end],       'train')
    copy_split(all_images[train_end:val_end], 'val')
    copy_split(all_images[val_end:],          'test')

    # Generate data.yaml — YOLO needs this file to know class names and data paths
    # Using dictionary format (0: sec1, 2: sec2, ...) to safely handle the skipped class ID 1
    names_yaml   = "\n".join([f"  {class_id}: {name}" for name, class_id in CLASS_MAPPING.items()])
    yaml_content = f"train: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names_yaml}\n"

    with open(os.path.join(dest_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"  ✅  Dataset ready!")
    print(f"  🏋️  Train : {train_end} images")
    print(f"  🔍  Val   : {val_end - train_end} images")
    print(f"  🧪  Test  : {total - val_end} images")
    print(f"  📄  data.yaml generated at: {dest_dir}/data.yaml")


# ==========================================
# MAIN EXECUTION
# This is where the script starts when you run: python3 main.py
# ==========================================

def main():
    # Read the list of stores and cameras from cameras.json
    with open(JSON_FILE) as f:
        data = json.load(f)

    # Create a shared dictionary so all camera processes can report their results back
    manager     = multiprocessing.Manager()
    return_dict = manager.dict()
    processes   = []

    # Start one background process for every camera defined in cameras.json
    # All cameras run at the same time (in parallel)
    for site in data:
        for cam in site["cameras"]:
            p = multiprocessing.Process(
                target=process_camera,
                args=(site["site_name"], cam["camera_id"], cam["rtsp_url"], return_dict)
            )
            p.start()
            processes.append(p)

    # Wait here until ALL camera processes finish collecting their images
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(1)
        print("\n" + "="*55)
        print("  🏁  ALL CAMERAS FINISHED — MAX_IMAGES reached for every camera.")
        print("="*55)
    except KeyboardInterrupt:
        # If you press Ctrl+C, all cameras are stopped cleanly
        print("\n  🛑  STOPPED — Cameras terminated by user (Ctrl+C).")
        for p in processes:
            p.terminate()

    # Print a summary table showing how each camera performed
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

    # Finally, package all annotated images into the training_dataset/ folder
    create_training_dataset()


# Python entry point — this block only runs when you execute the file directly
if __name__ == "__main__":
    main()