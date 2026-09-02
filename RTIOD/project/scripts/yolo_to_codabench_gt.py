import os
import glob
import json
from PIL import Image
from tqdm import tqdm

def yolo_to_codabench(yolo_val_img_dir, yolo_val_label_dir, output_json):
    img_paths = glob.glob(os.path.join(yolo_val_img_dir, '**', '*.png'), recursive=True)
    img_paths += glob.glob(os.path.join(yolo_val_img_dir, '**', '*.jpg'), recursive=True)
    
    gt_dict = {}
    
    for img_path in tqdm(img_paths, desc="Converting GT"):
        img_name = os.path.basename(img_path)
        uid = os.path.splitext(img_name)[0]
        
        # Initialize dictionary for this uid
        gt_dict[uid] = {'boxes': [], 'labels': []}
        
        label_path = os.path.join(yolo_val_label_dir, f"{uid}.txt")
        
        if not os.path.exists(label_path):
            continue
            
        # Get image dimensions to denormalize YOLO boxes
        with Image.open(img_path) as img:
            w, h = img.size
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls_id = int(parts[0])
            x_center, y_center, bw, bh = map(float, parts[1:5])
            
            # YOLO format is normalized [x_center, y_center, width, height]
            # Convert to absolute [x1, y1, x2, y2]
            abs_w = bw * w
            abs_h = bh * h
            abs_xc = x_center * w
            abs_yc = y_center * h
            
            x1 = abs_xc - (abs_w / 2)
            y1 = abs_yc - (abs_h / 2)
            x2 = abs_xc + (abs_w / 2)
            y2 = abs_yc + (abs_h / 2)
            
            # Note: CodaBench evaluator usually expects 1-indexed classes for COCO, but wait:
            # Let's check convert_gt_format.py. It used raw category_id. We'll use cls_id + 1 if YOLO is 0-indexed.
            # In detect.py, they did `labels = labels.int() + 1`. So we should do cls_id + 1 here.
            
            gt_dict[uid]['boxes'].append([x1, y1, x2, y2])
            gt_dict[uid]['labels'].append(cls_id + 1)
            
    with open(output_json, 'w') as f:
        json.dump(gt_dict, f, indent=4)
        
    print(f"Created ground-truth JSON for {len(gt_dict)} validation images at {output_json}")

if __name__ == '__main__':
    yolo_to_codabench(
        'data/pca_images/val',
        'data/pca_images/val', # YOLO labels are next to images in this setup? Actually let's pass explicitly
        'valid_targets_yolo.json'
    )
