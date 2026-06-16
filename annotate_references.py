import os
import cv2
from ultralytics import YOLO

# Custom integer mapping exactly as requested
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

REFERENCE_DIR = "reference_data"

def process_references():
    model = YOLO("yolov8n.pt")
    
    if not os.path.exists(REFERENCE_DIR):
        print(f"[ERROR] '{REFERENCE_DIR}' folder not found.")
        return

    total_images = 0
    total_persons = 0
    
    for folder_name, class_id in CLASS_MAPPING.items():
        folder_path = os.path.join(REFERENCE_DIR, folder_name)
        
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Skipping missing folder: {folder_name}")
            continue
            
        print(f"\nProcessing section: {folder_name} (Assigned Class ID: {class_id})")
        
        files_found = False
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            files_found = True
            image_path = os.path.join(folder_path, filename)
            
            # Load the image using OpenCV to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                print(f"  [ERROR] Could not read image {filename}")
                continue
                
            img_h, img_w = image.shape[:2]
            
            # Run YOLO inference
            results = model(image_path, classes=[0], conf=0.5, verbose=False)
            
            person_count = 0
            lines = []
            
            # Extract bounding boxes
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:  # If it is a person
                    # Calculate YOLO format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    xc = ((x1 + x2) / 2) / img_w
                    yc = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    
                    # OVERRIDE the class ID with the custom integer!
                    lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    person_count += 1
            
            if person_count > 0:
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(folder_path, txt_filename)
                
                with open(txt_path, "w") as f:
                    f.write("\n".join(lines))
                    
                total_images += 1
                total_persons += person_count
                print(f"  [ANNOTATED] {filename} -> Found {person_count} persons -> Wrote class '{class_id}'")
            else:
                print(f"  [SKIP] {filename} -> No persons found.")
                
        if not files_found:
            print("  (Folder is empty)")
                
    print("\n" + "="*40)
    print(" REFERENCE ANNOTATION SUMMARY ")
    print("="*40)
    print(f"Total reference images annotated: {total_images}")
    print(f"Total persons bounded: {total_persons}")
    print("="*40)
    print("\nReady for custom model training!")

if __name__ == "__main__":
    print(f"Starting Offline Reference Annotation script...")
    print(f"Target directory: {REFERENCE_DIR}/")
    process_references()
