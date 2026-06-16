import cv2
import os
import json
import time
import multiprocessing
from datetime import datetime
from ultralytics import YOLO

JSON_FILE = "cameras.json"
FRAME_INTERVAL = 2
BLUR_THRESHOLD = 50
MAX_IMAGES = 10

model = YOLO("yolov8n.pt")

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def is_blur(image, threshold=50):
    small = cv2.resize(image, (320, 240))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, score < threshold

def save_yolo_annotation(results, txt_file, img_w, img_h):
    lines = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(txt_file, "w") as f:
        f.write("\n".join(lines))

def process_camera(site_name, camera_id, rtsp_url, return_dict):
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
            # annotated = results[0].plot()
            # cv2.imwrite(f"{ann_img_folder}/{name}", annotated)
            cv2.imwrite(f"{ann_img_folder}/{name}", frame)
            txt_name = name.replace(".jpg", ".txt")
            save_yolo_annotation(results, f"{ann_txt_folder}/{txt_name}", frame.shape[1], frame.shape[0])
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