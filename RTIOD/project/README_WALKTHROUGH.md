# Rank 1 WACV Methodology Walkthrough

All files requested for the Rank 1 pipeline have been generated and directly injected cleanly into the `RTIOD/startingkit` framework directory!

## File Map Created
```bash
RTIOD/startingkit/
├── scripts/
│   ├── build_pca_dataset.py       # (Task 1)
│   ├── oversample_dataset.py      # (Task 2)
│   └── detect_rank1.py            # (Task 4)
├── configs/
│   └── rank1_yolov8m.yaml         # (Task 3a)
└── train_rank1.py                 # (Task 3b)
```

---

## 1. Phase 1: Data Preparation
We've replaced the naive 1-channel thermal inputs with the paper's High-Performance $[I_t, B_t, R_t]$ tensor generation.

**Execute Class Oversampling First:**
```bash
conda run -n thermal python scripts/oversample_dataset.py
```
> [!IMPORTANT]
> This step MUST be run before PCA. It will duplicate Bicycle annotations (2x total) and Motorcycle annotations (5x total) over the base 1-channel images and labels to balance the classes.

> [!TIP]
> **Performance:** `build_pca_dataset.py` explicitly utilizes PyTorch's native `.pca_lowrank` algorithm across a `multiprocessing` pool targeting all 6 of your DGX V100 GPUs! Ensure no other massive workload is occupying CUDA memory before launching this.

**Execute PCA Initialization Second:**
```bash
conda run -n thermal python scripts/build_pca_dataset.py
```
This will strip bounded objects dynamically and assemble new 3-channel images down into `data/pca_images/` covering both the original and newly oversampled minority frames!

---

## Phase 2: Distributed Training
`train_rank1.py` pulls from `configs/rank1_yolov8m.yaml` which embeds the new P2 Head (Stride 4) structure to capture distant pedestrians on highways!

The training constraints:
* **Resolution:** 288x384 (Using `rect=True` formatting).
* **Augmentation Restrictions:** Mosaic and layout-altering shifts have successfully been clamped to zero. Color variants (HSV Jitter) and horizontal flipping persist.
* **Architecture:** Initialized from `yolov8m.pt`. Distributed Data Parallel wraps directly to your array via `device="0,1,2,3,4,5"`.

**Execute Training:**
```bash
conda run -n thermal python train_rank1.py
```

---

## Phase 3: Spatial Predictions & Metrics
Once weights pop into `runs/detect/rank1_train/weights/best.pt`, we enforce the post-processing geometries.

`scripts/detect_rank1.py` precomputes the `scipy.spatial.ConvexHull` on the distribution of points in `data/labels/train/`. By mapping Valid Drivable Area mathematically, it seamlessly discards False Positives hitting the sky or tree canopies.

**Execute Inference:**
```bash
conda run -n thermal python scripts/detect_rank1.py
```
This will dump your filtered `predictions_rank1.json`. From there, pass it directly into the GPU accelerated evaluator I built for you previously to see your Rank 1 scores!
