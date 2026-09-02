import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def create_subset():
    base_dir = Path(__file__).resolve().parent.parent / "data"
    src_img_dir = base_dir / "pca_images" / "train"
    dst_img_dir = base_dir / "pca_images_10k" / "train"
    
    os.makedirs(dst_img_dir, exist_ok=True)
    
    print("Scanning for available image-label pairs...")
    all_imgs = list(src_img_dir.glob("*.png"))
    
    # Ensure we only pick images that have a corresponding .txt label in the same folder
    valid_pairs = [img for img in all_imgs if img.with_suffix('.txt').exists()]
    print(f"Found {len(valid_pairs)} completely valid pairs.")
    
    k = min(10000, len(valid_pairs))
    print(f"Randomly selecting {k} pairs...")
    subset = random.sample(valid_pairs, k)
    
    print(f"Copying files to {dst_img_dir}...")
    for img_path in tqdm(subset):
        lbl_path = img_path.with_suffix('.txt')
        
        # Copy image and label
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        shutil.copy2(lbl_path, dst_img_dir / lbl_path.name)
        
    print("Done! Your 10k dataset is ready.")

if __name__ == "__main__":
    create_subset()
