import os
import glob
from collections import defaultdict

LABELS_DIR = "/raid/ai24mtech12009/cv_proj/RTIOD/startingkit/data/labels/train"

files = glob.glob(os.path.join(LABELS_DIR, "**", "*.txt"), recursive=True)

before_counts = {0: 0, 1: 0, 2: 0, 3: 0}
after_counts = {0: 0, 1: 0, 2: 0, 3: 0}

for file in files:
    is_dup = "_dup" in os.path.basename(file)
    try:
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in after_counts:
                    after_counts[cls_id] += 1
                    if not is_dup:
                        before_counts[cls_id] += 1
    except Exception as e:
        continue

def print_stats(counts, label):
    total = sum(counts.values())
    if total == 0:
        print(f"{label}: No annotations found.")
        return
    print(f"--- {label} ---")
    print(f"Total Instances: {total}")
    names = {0: 'Person', 1: 'Bicycle', 2: 'Motorcycle', 3: 'Vehicle'}
    for k in sorted(counts.keys()):
        perc = (counts[k] / total) * 100
        print(f"{names[k]:>10} (Class {k}): {counts[k]:>8} ({perc:.2f}%)")
    print()

print_stats(before_counts, "BEFORE OVERSAMPLING")
print_stats(after_counts, "AFTER OVERSAMPLING") 
