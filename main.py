"""
VISION AI PIPELINE SCRIPT
=========================
This script is an automated tool that connects to security cameras and builds a dataset for AI training.

Here is exactly what it does, step by step:
1. It reads a list of camera links from 'cameras.json'.
2. It connects to these cameras and takes screenshots (frames).
3. Quality Check: It automatically throws away any blurry, too dark, or too bright images.
4. AI Person Detection: It uses a smart AI called 'YOLO' to find where the people are in the good images.
5. AI Clothing Matcher: It cuts out the picture of each person and uses another AI (ResNet) to look at their clothes. It compares their clothes to the reference pictures you provided to figure out what section they belong to (like 'sec1', 'sec2', or 'customers').
6. Saving Labels: It draws a box around the person, tags them with the correct section name, and saves this information.
7. Dataset Building: Finally, it takes all the good, perfectly labeled images and automatically splits them into 'train', 'val', and 'test' folders so that you can easily train your own custom YOLO model later!
"""
import os
import cv2
import json
import time
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
JSON_FILE = "cameras.json"
FRAME_INTERVAL = 2
BLUR_THRESHOLD = 150
MAX_IMAGES = 10

CLASS_MAPPING = {
    "sec1": 0,
    "sec2": 2,
    "sec3": 3,
    "sec4": 4,
    "sec5": 5,
    "sec6": 6,
    "sec7": 7,
    "customers": 8,
    "sec8": 9,
    "sec9": 10
}


# ==========================================
# UTILITIES
# ==========================================
def create_folder(path):
    """Safely create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def is_good_frame(image):
    """
    Analyzes an image for blur, darkness, overexposure, and low detail.
    Returns a dictionary of scores and a boolean indicating if it is good.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (400, 300))

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()

    # Rules for discarding bad frames
    is_blur = blur_score < 60
    is_dark = brightness < 40
    is_bright = brightness > 220
    is_low_detail = edge_density < 5

    is_bad = is_blur or is_dark or is_bright or is_low_detail

    score = {
        "blur_score": blur_score,
        "brightness": brightness,
        "edge_density": edge_density
    }

    return score, not is_bad


# ==========================================
# REFERENCE MATCHER (DEEP LEARNING)
# ==========================================
class ReferenceMatcher:
    """
    Uses ResNet18 to extract deep visual features from clothing.
    Matches detected people against reference_data images.
    """
    def __init__(self, ref_dir="reference_data", threshold=0.75):
        self.ref_dir = ref_dir
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load ResNet18 Feature Extractor safely
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
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
            
        # Strip classification head to output raw 512-d embeddings
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.reference_embeddings = []
        self._load_references()
        
    def _compute_embedding(self, image):
        """Pass image through ResNet to get a normalized feature vector."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        # Crop the inner 80% of bounding box to eliminate background noise but keep context
        width, height = pil_img.size
        if width > 0 and height > 0:
            left, top = width * 0.1, height * 0.1
            right, bottom = width * 0.9, height * 0.9
            pil_img = pil_img.crop((left, top, right, bottom))
        
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(input_tensor)
            
        return torch.nn.functional.normalize(embedding, p=2, dim=1)
        
    def _load_references(self):
        """Loads and precomputes embeddings for all reference_data images."""
        if not os.path.exists(self.ref_dir):
            return
            
        for folder_name, class_id in CLASS_MAPPING.items():
            folder_path = os.path.join(self.ref_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    emb = self._compute_embedding(image)
                    self.reference_embeddings.append((emb, class_id))
                    
    def match(self, cropped_img):
        """Compares a cropped person to references using Cosine Similarity."""
        if not self.reference_embeddings:
            return CLASS_MAPPING.get("customers", 8)
            
        crop_emb = self._compute_embedding(cropped_img)
        best_score = -1.0
        best_class = CLASS_MAPPING.get("customers", 8)
        
        for ref_emb, class_id in self.reference_embeddings:
            score = torch.sum(crop_emb * ref_emb).item()
            if score > best_score:
                best_score = score
                if score > self.threshold:
                    best_class = class_id
                    
        return best_class


# ==========================================
# PROCESSING LOGIC
# ==========================================
def save_yolo_annotation(results, txt_file, image, matcher, crop_folder=None, base_name=""):
    """Saves YOLO labels, crops persons into section folders, and draws bounding boxes."""
    img_h, img_w = image.shape[:2]
    lines = []
    annotated_image = image.copy()
    ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
    
    for idx, box in enumerate(results[0].boxes):
        if int(box.cls[0]) == 0:  # If person
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
            x2_c, y2_c = min(img_w, int(x2)), min(img_h, int(y2))
            cropped = image[y1_c:y2_c, x1_c:x2_c]
            
            # Match person to section
            class_id = CLASS_MAPPING.get("customers", 8) if cropped.size == 0 else matcher.match(cropped)
                
            # YOLO format coords
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            
            # Draw annotation
            label = ID_TO_CLASS.get(class_id, "unknown")
            cv2.rectangle(annotated_image, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1_c, max(0, y1_c - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Save section crop
            if crop_folder and base_name and cropped.size > 0:
                label_crop_folder = os.path.join(crop_folder, label)
                create_folder(label_crop_folder)
                cv2.imwrite(os.path.join(label_crop_folder, f"{idx}_{base_name}"), cropped)
            
    with open(txt_file, "w") as f:
        f.write("\n".join(lines))
        
    return annotated_image


def process_camera(site_name, camera_id, rtsp_url, return_dict):
    """Main loop for a single camera process."""
    # Instantiate models inside process to ensure safe multiprocessing
    matcher = ReferenceMatcher()
    model = YOLO("yolov8s.pt") # Upgraded to 'small' model for better CCTV detection
    
    return_dict[camera_id] = {"site_name": site_name, "clear": 0, "blur": 0, "annotated": 0, "persons": 0}
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[ERROR] {camera_id}")
        stats = return_dict[camera_id]
        stats["error"] = "Connection failed (Camera offline or unreachable)"
        return_dict[camera_id] = stats
        return
        
    print(f"[STARTED] {camera_id}")
    count = 0
    
    while count < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print(f"[RECONNECT] {camera_id}")
            time.sleep(5)
            continue
            
        # Folder structure
        date = datetime.now().strftime("%Y-%m-%d")
        base = f"dataset/{date}/{site_name}"
        image_folder, blur_folder = f"{base}/images", f"{base}/blur"
        ann_img_folder, ann_txt_folder = f"{base}/annotations/images", f"{base}/annotations/txt"
        crop_folder = f"{base}/crops"
        
        for folder in [image_folder, blur_folder, ann_img_folder, ann_txt_folder, crop_folder]:
            create_folder(folder)
            
        # Quality check
        score_dict, is_good = is_good_frame(frame)
        name = f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        if not is_good:
            cv2.imwrite(f"{blur_folder}/{name}", frame)
            status = "BLUR"
            stats = return_dict[camera_id]
            stats["blur"] += 1
            return_dict[camera_id] = stats
        else:
            image_path = f"{image_folder}/{name}"
            cv2.imwrite(image_path, frame)
            
            # Detect (Lowered confidence to 0.25 to catch more people)
            results = model(image_path, classes=[0], conf=0.25, verbose=False)
            txt_name = name.replace(".jpg", ".txt")
            
            # Annotate & Crop
            annotated = save_yolo_annotation(results, f"{ann_txt_folder}/{txt_name}", frame, matcher, crop_folder, name)
            cv2.imwrite(f"{ann_img_folder}/{name}", annotated)
            status = "CLEAR"
            
            # Update Stats
            person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
            stats = return_dict[camera_id]
            stats["clear"] += 1
            if person_count > 0:
                stats["annotated"] += 1
                stats["persons"] += person_count
            return_dict[camera_id] = stats
            
        count += 1
        score_str = f"Blr:{score_dict['blur_score']:.1f} Brt:{score_dict['brightness']:.1f} Edg:{score_dict['edge_density']:.1f}"
        print(f"[{status}] {camera_id} {score_str} {count}/{MAX_IMAGES}")
        time.sleep(FRAME_INTERVAL)
        
    cap.release()


# ==========================================
# DATASET GENERATOR
# ==========================================
def create_training_dataset(source_dir="dataset", dest_dir="training_dataset", split_ratios=(0.7, 0.2, 0.1)):
    """Compiles clear, annotated images into a perfectly formatted YOLO dataset."""
    print(f"\nCreating final YOLO dataset structure in '{dest_dir}'...")
    
    for split in ['train', 'val', 'test']:
        create_folder(f"{dest_dir}/images/{split}")
        create_folder(f"{dest_dir}/labels/{split}")
        
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found.")
        return
        
    all_images = []
    
    for date_folder in os.listdir(source_dir):
        date_path = os.path.join(source_dir, date_folder)
        if not os.path.isdir(date_path): continue
            
        for site_folder in os.listdir(date_path):
            site_path = os.path.join(date_path, site_folder)
            if not os.path.isdir(site_path): continue
                
            img_dir = os.path.join(site_path, "images")
            txt_dir = os.path.join(site_path, "annotations", "txt")
            
            if not os.path.exists(img_dir) or not os.path.exists(txt_dir): continue
                
            for img_name in os.listdir(img_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                    
                txt_name = img_name.rsplit('.', 1)[0] + ".txt"
                img_path, txt_path = os.path.join(img_dir, img_name), os.path.join(txt_dir, txt_name)
                
                # Only include perfectly annotated frames
                if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                    all_images.append((img_path, txt_path, img_name, txt_name))
                    
    if not all_images:
        print("No annotated images found to create training dataset.")
        return
        
    random.seed(42)
    random.shuffle(all_images)
    
    total = len(all_images)
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])
    
    def copy_split(data, split_name):
        for img_path, txt_path, img_name, txt_name in data:
            shutil.copy(img_path, os.path.join(dest_dir, "images", split_name, img_name))
            shutil.copy(txt_path, os.path.join(dest_dir, "labels", split_name, txt_name))
            
    copy_split(all_images[:train_end], 'train')
    copy_split(all_images[train_end:val_end], 'val')
    copy_split(all_images[val_end:], 'test')
    
    # Generate data.yaml mapping (Dictionary format handles skipped index 1)
    names_yaml = "\n".join([f"  {class_id}: {name}" for name, class_id in CLASS_MAPPING.items()])
    yaml_content = f"train: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names_yaml}\n"
    
    with open(os.path.join(dest_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)
         
    print(f"Dataset created successfully!")
    print(f"Train: {train_end} | Val: {val_end - train_end} | Test: {total - val_end}")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    with open(JSON_FILE) as f:
        data = json.load(f)
        
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    
    for site in data:
        for cam in site["cameras"]:
            p = multiprocessing.Process(target=process_camera, args=(site["site_name"], cam["camera_id"], cam["rtsp_url"], return_dict))
            p.start()
            processes.append(p)
            
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(1)
        print("\n[DONE] All cameras reached MAX_IMAGES and finished.")
    except KeyboardInterrupt:
        print("\n[STOP] Terminating cameras...")
        for p in processes:
            p.terminate()
            
    print("\n" + "="*40)
    print(" RUN SUMMARY ")
    print("="*40)
    
    total_clear = total_blur = total_anno = total_persons = 0
    
    for cam_id, stats in return_dict.items():
        print(f"Site: {stats['site_name']} | Camera: {cam_id}")
        if "error" in stats:
            print(f"  - Status: ERROR ({stats['error']})")
        print(f"  - Clear images:     {stats['clear']}")
        print(f"  - Blur images:      {stats['blur']}")
        print(f"  - Annotated images: {stats['annotated']}")
        print(f"  - Total persons:    {stats['persons']}")
        
        total_clear += stats.get('clear', 0)
        total_blur += stats.get('blur', 0)
        total_anno += stats.get('annotated', 0)
        total_persons += stats.get('persons', 0)
        
    print("-" * 40)
    print(f"Total Clear Images:     {total_clear}")
    print(f"Total Blur Images:      {total_blur}")
    print(f"Total Annotated Images: {total_anno}")
    print(f"Total Persons Found:    {total_persons}")
    print("="*40)
    
    create_training_dataset()


if __name__ == "__main__":
    main()