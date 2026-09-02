import os
import glob
import json
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
from ultralytics import YOLO
from tqdm import tqdm

LABELS_DIR = "data/labels/train"
VAL_DIR = "data/pca_images/val"
OUTPUT_JSON = "submissions/predictions_v12m.json" 

def build_convex_hull():
    print("Building global convex hull from training annotations...")
    all_points = [] 
    
    label_files = glob.glob(os.path.join(LABELS_DIR, "**", "*.txt"), recursive=True)
    for lbl in label_files:
        with open(lbl, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                # YOLO format: cls x_c y_c w h
                x_c, y_c = float(parts[1]), float(parts[2])
                all_points.append([x_c, y_c])
                
    points_arr = np.array(all_points)
    hull = ConvexHull(points_arr)
    # Get the vertices to build a polygon path for point-in-polygon tests
    hull_path = MplPath(points_arr[hull.vertices])
    print(f"Hull built tightly mapping {len(points_arr)} ground-truth targets.")
    return hull_path

def detect_and_filter(hull_path):
    print("Loading model...")
    model = YOLO("runs/detect/ablation_yolov12m_10k/weights/best.pt")
    
    print("Running inference and applying spatial hull filter...")
    val_images = glob.glob(os.path.join(VAL_DIR, "**", "*.jpg"), recursive=True)
    val_images += glob.glob(os.path.join(VAL_DIR, "**", "*.png"), recursive=True)
    
    submission = {}
    
    filtered_count = 0
    total_preds = 0
    for img_path in tqdm(val_images, desc="Inference"):
        img_name = os.path.basename(img_path)
        uid = os.path.splitext(img_name)[0]
        
        # Inference using rectangular evaluation matching training
        results = model.predict(source=img_path, imgsz=[288, 384], augment=False, conf=0.01, verbose=False)
        
        boxes = []
        scores = []
        labels = []
        
        if len(results) > 0:
            res = results[0]
            # Use cpu side arrays
            boxes_norm = res.boxes.xywhn.cpu().numpy()
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            
            for i in range(len(confs)):
                total_preds += 1
                x_c, y_c, w, h = boxes_norm[i]
                
                # Sptial Constraint Evaluation
                # if not hull_path.contains_point((x_c, y_c)):
                #     # Prediction center lies in the sky or deep vegetation beyond valid drivable area
                #     filtered_count += 1
                #     continue
                    
                x1, y1, x2, y2 = boxes_xyxy[i]
                
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                scores.append(float(confs[i]))
                # Convert back to COCO 1-indexed taxonomy
                labels.append(int(cls[i]) + 1)
                
        submission[uid] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }
        
    print(f"\nSpatial Hull effectively discarded {filtered_count} / {total_preds} False Positives.")
    print(f"Saving filtered predictions to {OUTPUT_JSON}")
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(submission, f, indent=4)
        
if __name__ == '__main__':
    hull_mask = build_convex_hull()
    detect_and_filter(hull_mask)
