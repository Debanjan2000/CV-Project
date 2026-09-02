from pathlib import Path

# path = Path("data/images/val")
path = Path("RTIOD/startingkit/data/pca_images/train") 
# path = Path("RTIOD/startingkit/data/labels/train")
# Count only files (excluding directories)
file_count = len([f for f in path.iterdir() if f.is_file()])
print(f"Number of files: {file_count}")
