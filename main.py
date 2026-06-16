import cv2
import os
import json
import time
import multiprocessing
from datetime import datetime
from ultralytics import YOLO
import numpy as np

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
    def __init__(self, ref_dir="reference_data", threshold=0.6):
        self.ref_dir = ref_dir
        self.threshold = threshold
        self.reference_hists = []
        self._load_references()
        
    def _compute_hist(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist
        
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
                    hist = self._compute_hist(image)
                    self.reference_hists.append((hist, class_id))
                    
    def match(self, cropped_img):
        if not self.reference_hists:
            return CLASS_MAPPING.get("customers", 8)
            
        crop_hist = self._compute_hist(cropped_img)
        best_score = -1
        best_class = CLASS_MAPPING.get("customers", 8)
        
        for ref_hist, class_id in self.reference_hists:
            score = cv2.compareHist(crop_hist, ref_hist, cv2.HISTCMP_CORREL)
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

def is_blur(image, threshold=50):
    small = cv2.resize(image, (320, 240))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, score < threshold

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
        score, blur = is_blur(frame, BLUR_THRESHOLD)
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
        print(f"[{status}] {camera_id} Score:{score:.2f} {count}/{MAX_IMAGES}")
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

if __name__ == "__main__":
    main()