# RTIOD WACV Rank-1 Execution Summary & Agent Handoff

This document summarizes the pipeline, code architecture, and key engineering learnings derived from implementing the Rank-1 approach for the RTIOD Challenge. It is designed to rapidly onboard a new autonomous agent so they can seamlessly continue development or debugging.

---

## 1. Context & Objective
The goal is to reproduce the **WACV Rank-1 solution** for Thermal Object Detection on the RTIOD dataset. Thermal imaging often suffers from low contrast and heavy background thermal noise (e.g., sun-heated roads). The Rank-1 approach utilizes **Two-Pass Global PCA Background Decomposition** to separate moving objects from a static background, generating 3-channel (RGB-like) tensor inputs for YOLO models.

## 2. Methodology: PCA-based Decomposition

### Pass 1: Background Template Generation
1. **Temporal Grouping**: Frames are grouped by 1-hour temporal bins.
2. **Masking:** Using ground-truth YOLO labels, objects (cars, pedestrians, bikes) are precisely masked out.
3. **Temporal Averaging:** We compute the mean of unmasked pixels over the 1-hour bin, yielding a purely empty "clean background" image ($\bar{I}_{g}$) for that group. Groups with complete occlusion of any pixel (0 valid contributions) are discarded.
4. **Subspace Learning:** We treat the valid 1,225 background templates as vectors and run SVD/PCA (`torch.pca_lowrank`) to extract the **top $K=50$ principal components** ($U$) and the global mean ($\mu$).
5. **Output**: A PyTorch dictionary `pca_model.pt` containing $U$ and $\mu$.

### Pass 2: 3-Channel Tensor Generation
For every real frame $I_t$:
1. **Background Reconstruction ($B_t$)**: Project the frame onto the PCA subspace ($U @ U^T @ (I_t - \mu) + \mu$) to get a synthesized clean background image representing current ambient thermal conditions.
2. **Residual Calculation ($R_t$)**: $R_t = I_t - B_t$. This highlights objects while cancelling out the static background.
3. **Normalization**: To match the paper's visuals exactly (dark background, bright glowing objects), $R_t$ uses **per-image MIN-MAX Absolute scaling** (`abs()` $\rightarrow$ min-max to `[0, 255]`), NOT a fixed $+128$ shift.
4. **Merging**: The final image is constructed as `X_t = cv2.merge([R_t, B_t, I_t])` representing (BGR array mapping). Red channel is original, Green is background, Blue is residual. This causes empty roads to appear gold/yellow and objects to glow cyan/magenta.

---

## 3. Core Architecture & Files

The pipeline operates out of `/raid/ai24mtech12009/cv_proj/RTIOD/startingkit/`:

| Component | Path | Description |
| :--- | :--- | :--- |
| **Data Config** | `data/data_pca.yaml` | YAML file defining the YOLO dataset paths (needs exact symmetric `/images/` and `/labels/` nested structure). |
| **Model Config** | `configs/rank1_yolov8m.yaml` | A heavily modified YOLOv8m config incorporating a **P2 Head (Stride 4)** for detecting intensely small pedestrians at long ranges. |
| **Oversampling** | `scripts/oversample_dataset.py` | Runs first. Duplicates images with Bicycles (2x) and Motorcycles (5x) to tackle dataset imbalance. |
| **PCA Builder** | `scripts/build_pca_dataset.py` | The absolute workhorse. Executes Pass 1 and Pass 2 across `multiprocessing` CPU workers and batched GPU logic. |
| **PCA Debugger**| `scripts/debug_pca.py` | Lightweight script that runs on 8 sampled images to visibly verify the math is dropping out backgrounds correctly. |
| **Trainer** | `train_rank1.py` | YOLOv8 `model.train()` execution on 6x DGX V100s, enforcing strict augment restrictions (no zoom/mosaic). |
| **Detector** | `scripts/detect_rank1.py` | Uses `scipy.spatial.ConvexHull` on the ground truth to mask out the sky/trees from predictions and eradicate False Positives. |

---

## 4. Hardware Optimizations & Critical Lessons Learned

If you are an agent modifying this code, **READ THESE TRAPS:**

1. **GPU Starvation (PCIe Bottleneck)**: 
   Initially, `build_pca_dataset.py` sent frames into the GPU one-by-one. $110592 \times 50$ matrix multiplications finish in microseconds on V100 Tensor Cores, leading to massive PCIe overhead and 0% GPU utilization. **Fix:** Tensor batching (Batch size = 256) was implemented. We accumulate 256 frames into a `(d, 256)` matrix on the CPU first, push to GPU *once*, compute all at once, and pull back *once*. This is non-negotiable for speed.

2. **Residual Dynamic Range Washes Out**: 
   When calculating $R_t$, simple offsets like `R_t + 128` look gray and confusing to the network. You MUST apply `np.abs(R_t)` before min-max scaling to stretch the dynamic anomaly range correctly.

3. **Storage/Inode Limits**: 
   Writing 438,422 3-channel PNGs requires massive disk space ($\approx$ 70 GB). Attempting to keep the original raw frames hierarchy plus the flat copy plus the PCA copy blew out disk quotas. Redundant `data/frames/` had to be deleted.
   *Note: If the script dies midway, it is designed with `os.path.exists()` checks so workers will seamlessly resume on missing files.*

4. **YOLO Dataset Path Mapping Issue**: 
   YOLO is rigidly hardcoded to find labels by substituting the string `images` with `labels` in the filepath. 
   If your image path is `data/pca_images/train/`, YOLO will aggressively look for `data/pca_labels/train/`. 
   **Fix:** We must symlink the original YOLO label directories to align structurally with wherever the PCA images are stored, e.g., `ln -s data/labels data/pca_dataset/labels` and `ln -s data/pca_images data/pca_dataset/images`.

5. **`_dup` File Over-Isolation**: 
   When splitting string titles to gather clip names (e.g., `frames_2020..._image_0000_dup1.jpg`), simple `_` splits broke temporal associations. It is strictly required to split entirely by `_image_` to group augmented frames with their original temporal clips correctly during Pass 1.
