import os
import glob
import shutil
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR = str(BASE_DIR.parent / "data" / "images" / "train")
LABELS_DIR = str(BASE_DIR.parent / "data" / "labels" / "train")

def oversample_minority_classes():
    """
    Oversamples minority classes to fix class imbalance:
    Class 1 (Bicycle) -> 2x Total (1 duplication)
    Class 2 (Motorcycle) -> 5x Total (4 duplications)
    """
    print("Initiating Minority Class Oversampling...")
    all_labels = glob.glob(os.path.join(LABELS_DIR, "**", "*.txt"), recursive=True)
    
    dups_created = 0
    
    for lbl_path in tqdm(all_labels, desc="Oversampling"):
        # We only want to process original files, not already duplicated ones
        if '_dup' in lbl_path:
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        has_bicycle = False
        has_motorcycle = False
        
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            cls_id = int(parts[0])
            if cls_id == 1:
                has_bicycle = True
            if cls_id == 2:
                has_motorcycle = True
                
        duplications = 0
        if has_motorcycle:
            duplications = 4
        elif has_bicycle:
            duplications = 1
            
        if duplications > 0:
            img_path_png = str(lbl_path).replace(LABELS_DIR, IMAGES_DIR).replace('.txt', '.png')
            img_path_jpg = str(lbl_path).replace(LABELS_DIR, IMAGES_DIR).replace('.txt', '.jpg')
            
            img_target = img_path_png if os.path.exists(img_path_png) else img_path_jpg
            
            if not os.path.exists(img_target):
                continue
                
            base_img = os.path.splitext(img_target)[0]
            ext_img = os.path.splitext(img_target)[1]
            base_lbl = os.path.splitext(lbl_path)[0]
            ext_lbl = ".txt"
            
            for d in range(1, duplications + 1):
                new_img = f"{base_img}_dup{d}{ext_img}"
                new_lbl = f"{base_lbl}_dup{d}{ext_lbl}"
                
                shutil.copy2(img_target, new_img)
                shutil.copy2(lbl_path, new_lbl)
                dups_created += 1
                
    print(f"Oversampling Complete: Generated {dups_created} new duplicated samples.")

if __name__ == '__main__':
    oversample_minority_classes()
