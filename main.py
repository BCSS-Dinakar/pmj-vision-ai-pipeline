import cv2
import os
import json
import time
import multiprocessing
import shutil
import random
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

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

class ReferenceMatcher:
    def __init__(self, ref_dir="reference_data", threshold=0.85):
        self.ref_dir = ref_dir
        self.threshold = threshold
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
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
            
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.reference_embeddings = []
        self._load_references()
        
    def _compute_embedding(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        # Crop the center 60% of the bounding box to focus on torso and avoid background
        width, height = pil_img.size
        left = width * 0.2
        top = height * 0.2
        right = width * 0.8
        bottom = height * 0.8
        if width > 0 and height > 0:
            pil_img = pil_img.crop((left, top, right, bottom))
        
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(input_tensor)
            
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
        
    def _load_references(self):
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

JSON_FILE = "cameras.json"
FRAME_INTERVAL = 2
BLUR_THRESHOLD = 150
MAX_IMAGES = 10

model = YOLO("yolov8n.pt")

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def is_good_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (400, 300))

    # 1. Blur score (sharpness)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Brightness check
    brightness = gray.mean()

    # 3. Simple noise check (edge density)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()

    # RULES
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

def save_yolo_annotation(results, txt_file, image, matcher, crop_folder=None, base_name=""):
    img_h, img_w = image.shape[:2]
    lines = []
    annotated_image = image.copy()
    ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
    
    for idx, box in enumerate(results[0].boxes):
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
            x2_c, y2_c = min(img_w, int(x2)), min(img_h, int(y2))
            cropped = image[y1_c:y2_c, x1_c:x2_c]
            
            if cropped.size == 0:
                class_id = CLASS_MAPPING.get("customers", 8)
            else:
                class_id = matcher.match(cropped)
                
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            
            label = ID_TO_CLASS.get(class_id, "unknown")
            cv2.rectangle(annotated_image, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1_c, max(0, y1_c - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if crop_folder and base_name and cropped.size > 0:
                label_crop_folder = os.path.join(crop_folder, label)
                create_folder(label_crop_folder)
                crop_path = os.path.join(label_crop_folder, f"{idx}_{base_name}")
                cv2.imwrite(crop_path, cropped)
            
    with open(txt_file, "w") as f:
        f.write("\n".join(lines))
        
    return annotated_image

def process_camera(site_name, camera_id, rtsp_url, return_dict):
    matcher = ReferenceMatcher()
    return_dict[camera_id] = {"site_name": site_name, "clear": 0, "blur": 0, "annotated": 0, "persons": 0}
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR]", camera_id)
        stats = return_dict[camera_id]
        stats["error"] = "Connection failed (Camera offline or unreachable)"
        return_dict[camera_id] = stats
        return
    print("[STARTED]", camera_id)
    count = 0
    while count < MAX_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("[RECONNECT]", camera_id)
            time.sleep(5)
            continue
        date = datetime.now().strftime("%Y-%m-%d")
        base = f"dataset/{date}/{site_name}"
        image_folder = f"{base}/images"
        blur_folder = f"{base}/blur"
        ann_img_folder = f"{base}/annotations/images"
        ann_txt_folder = f"{base}/annotations/txt"
        crop_folder = f"{base}/crops"
        for folder in [image_folder, blur_folder, ann_img_folder, ann_txt_folder, crop_folder]:
            create_folder(folder)
        score_dict, is_good = is_good_frame(frame)
        blur = not is_good
        name = f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        if blur:
            cv2.imwrite(f"{blur_folder}/{name}", frame)
            status = "BLUR"
            stats = return_dict[camera_id]
            stats["blur"] += 1
            return_dict[camera_id] = stats
        else:
            image_path = f"{image_folder}/{name}"
            cv2.imwrite(image_path, frame)
            results = model(image_path, classes=[0], conf=0.5, verbose=False)
            txt_name = name.replace(".jpg", ".txt")
            annotated = save_yolo_annotation(results, f"{ann_txt_folder}/{txt_name}", frame, matcher, crop_folder, name)
            cv2.imwrite(f"{ann_img_folder}/{name}", annotated)
            status = "CLEAR"
            
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
    total_clear = 0
    total_blur = 0
    total_anno = 0
    total_persons = 0
    
    for cam_id, stats in return_dict.items():
        print(f"Site: {stats['site_name']} | Camera: {cam_id}")
        if "error" in stats:
            print(f"  - Status: ERROR ({stats['error']})")
        print(f"  - Clear images:     {stats['clear']}")
        print(f"  - Blur images:      {stats['blur']}")
        print(f"  - Annotated images: {stats['annotated']}")
        print(f"  - Total persons:    {stats['persons']}")
        total_clear += stats['clear']
        total_blur += stats['blur']
        total_anno += stats['annotated']
        total_persons += stats['persons']
        
    print("-" * 40)
    print(f"Total Clear Images:     {total_clear}")
    print(f"Total Blur Images:      {total_blur}")
    print(f"Total Annotated Images: {total_anno}")
    print(f"Total Persons Found:    {total_persons}")
    print("="*40)
    
    create_training_dataset()

def create_training_dataset(source_dir="dataset", dest_dir="training_dataset", split_ratios=(0.7, 0.2, 0.1)):
    print(f"\nCreating final YOLO dataset structure in '{dest_dir}'...")
    
    for split in ['train', 'val', 'test']:
        create_folder(f"{dest_dir}/images/{split}")
        create_folder(f"{dest_dir}/labels/{split}")
        
    all_images = []
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found.")
        return
        
    for date_folder in os.listdir(source_dir):
        date_path = os.path.join(source_dir, date_folder)
        if not os.path.isdir(date_path): continue
            
        for site_folder in os.listdir(date_path):
            site_path = os.path.join(date_path, site_folder)
            if not os.path.isdir(site_path): continue
                
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
    
    train_data = all_images[:train_end]
    val_data = all_images[train_end:val_end]
    test_data = all_images[val_end:]
    
    def copy_split(data, split_name):
        for img_path, txt_path, img_name, txt_name in data:
            shutil.copy(img_path, os.path.join(dest_dir, "images", split_name, img_name))
            shutil.copy(txt_path, os.path.join(dest_dir, "labels", split_name, txt_name))
            
    copy_split(train_data, 'train')
    copy_split(val_data, 'val')
    copy_split(test_data, 'test')
    
    # Generate data.yaml automatically
    names_yaml = "\n".join([f"  {class_id}: {name}" for name, class_id in CLASS_MAPPING.items()])
        
    yaml_content = f"""train: images/train
val: images/val
test: images/test

names:
{names_yaml}
"""
    with open(os.path.join(dest_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)
         
    print(f"Dataset created successfully!")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    main()