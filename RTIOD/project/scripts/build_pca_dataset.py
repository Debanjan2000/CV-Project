"""
Two-Pass Global PCA Background-Residual Tensor Generator

Pass 1: Build Background Templates
  - Group frames by hourly timestamp (e.g. 20200514_13)
  - For each group g, compute masked background template I_bar_g
    using per-pixel averaging ONLY over unmasked (object-free) frames
  - Discard any group where ANY pixel has zero valid contributions
  - Flatten valid templates into matrix X ∈ R^{d×N}
  - Compute global mean μ and top-50 PCA basis U ∈ R^{d×50}

Pass 2: Apply Global PCA to Every Frame
  - For each frame I_t, project onto learned subspace:
      B_t = reshape(U @ U^T @ (vec(I_t) - μ) + μ)
      R_t = I_t - B_t
  - Save 3-channel tensor X_t = [I_t, B_t, R_t]
"""

import os
import glob
import re
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import pickle

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR.parent / "data" / "images"
LABELS_DIR = BASE_DIR.parent / "data" / "labels"
OUTPUT_DIR = BASE_DIR.parent / "data" / "pca_images"
PCA_MODEL_PATH = BASE_DIR.parent / "data" / "pca_model.pt"

# Image dimensions (fixed for RTIOD)
IMG_H, IMG_W = 288, 384
D = IMG_H * IMG_W  # 110592

# PCA rank
K = 50


# ─── Utility functions ──────────────────────────────────────────────

def get_hourly_group(filename):
    """
    Extract temporal group key = date + hour.
    e.g. frames_20200514_clip_0_1331_image_0000.jpg → '20200514_13'
    The 4-digit time field (1331) is after the clip number; first 2 digits = hour.
    """
    base = os.path.basename(filename)
    # Match: frames_YYYYMMDD_clip_<N>_HHMM_image_...
    m = re.match(r'frames_(\d{8})_clip_\d+_(\d{2})\d{2}_image_', base)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None


def get_original_name(filename):
    """
    Strip _dup<N> suffix to find the original label file.
    e.g. frames_..._image_0014_dup1.jpg → frames_..._image_0014.jpg
    """
    base = os.path.basename(filename)
    return re.sub(r'_dup\d+', '', base)


def load_mask(img_path, h, w):
    """
    Build the foreground mask M_{g,j} from YOLO labels.
    M = 1 inside bounding boxes (foreground), 0 elsewhere (background).
    Returns (1 - M) which is the background mask for element-wise multiplication.
    """
    orig_name = get_original_name(img_path)
    label_name = os.path.splitext(orig_name)[0] + '.txt'

    # Try both the direct path and the original (non-dup) path
    label_path = os.path.join(
        str(img_path).replace('images', 'labels').rsplit('/', 1)[0],
        label_name
    )

    bg_mask = np.ones((h, w), dtype=np.float32)  # 1 = background

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, x_c, y_c, bw, bh = map(float, parts[:5])
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    bg_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0.0

    return bg_mask


# ═══════════════════════════════════════════════════════════════════════
#  PASS 1: Build background templates and learn global PCA subspace
# ═══════════════════════════════════════════════════════════════════════

def build_template_for_group(args):
    """
    Worker function: builds one background template I_bar_g for a single
    hourly group using strict per-pixel averaging.

    Returns: (group_id, template_vector) or (group_id, None) if discarded.
    """
    group_id, img_paths = args

    cv2.setNumThreads(0)

    # Accumulators for per-pixel averaging
    sum_bg = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    count_bg = np.zeros((IMG_H, IMG_W), dtype=np.float64)

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        if h != IMG_H or w != IMG_W:
            img = cv2.resize(img, (IMG_W, IMG_H))

        bg_mask = load_mask(img_path, IMG_H, IMG_W)

        # Accumulate: only background pixels contribute
        # I_tilde_{g,j} = I_{g,j} * (1 - M_{g,j})   (bg_mask IS already 1-M)
        sum_bg += img.astype(np.float64) * bg_mask
        count_bg += bg_mask

    # Check the discard condition: if ANY pixel has zero valid contributions
    if np.any(count_bg == 0):
        return (group_id, None)

    # Per-pixel average: I_bar_g(p) = sum / count
    template = sum_bg / count_bg
    template_vec = template.flatten().astype(np.float32)

    return (group_id, template_vec)


def pass1_build_pca_model(split='train'):
    """
    Pass 1: Build all hourly background templates, fit global PCA.
    Saves: mean vector μ and top-K eigenvectors U to disk.
    """
    print("=" * 70)
    print("PASS 1: Building Background Templates & Learning Global PCA Subspace")
    print("=" * 70)

    images_dir = IMAGES_DIR / split
    all_images = glob.glob(str(images_dir / "*.jpg")) + \
                 glob.glob(str(images_dir / "*.png"))

    # ── Step 1: Group frames by hourly timestamp ──
    groups = {}
    skipped = 0
    for img in all_images:
        gid = get_hourly_group(img)
        if gid is None:
            skipped += 1
            continue
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(img)

    print(f"Total images: {len(all_images)}")
    print(f"Hourly groups found: {len(groups)}")
    if skipped > 0:
        print(f"Skipped (no parseable timestamp): {skipped}")

    # ── Step 2: Build templates in parallel (CPU-only, IO-bound) ──
    tasks = list(groups.items())
    num_workers = min(48, len(tasks))

    print(f"Building templates with {num_workers} workers...")
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(build_template_for_group, tasks),
            total=len(tasks),
            desc="Pass 1: Templates"
        ))

    # ── Step 3: Filter valid templates ──
    valid_templates = []
    discarded = 0
    for gid, vec in results:
        if vec is not None:
            valid_templates.append(vec)
        else:
            discarded += 1

    N = len(valid_templates)
    print(f"\nValid templates: {N}")
    print(f"Discarded groups (pixel always occluded): {discarded}")

    if N == 0:
        raise RuntimeError("No valid templates! Cannot fit PCA.")

    # ── Step 4: Compute global PCA on GPU ──
    print(f"\nFitting PCA with k={K} components on {N} templates (d={D})...")

    # X ∈ R^{d × N}  (each column is a flattened template)
    X = torch.from_numpy(np.stack(valid_templates, axis=0)).cuda(2)  # (N, d)
    X = X.t()  # → (d, N)

    # Mean background vector μ
    mu = X.mean(dim=1, keepdim=True)  # (d, 1)

    # Center the data
    X_centered = X - mu  # (d, N)

    # SVD: X_centered = U @ S @ V^T
    # We want top-K left singular vectors = top-K eigenvectors of covariance
    print("Running GPU SVD (torch.pca_lowrank)...")
    U, S, V = torch.pca_lowrank(X_centered.t(), q=K, center=False)
    # pca_lowrank expects (N, d) input and returns:
    #   U_lr: (N, K), S: (K,), V: (d, K)
    # V contains the principal components we need

    U_basis = V  # (d, K) — the projection basis

    # Verify orthonormality
    ortho_check = torch.mm(U_basis.t(), U_basis)
    print(f"Orthonormality check (should be identity): "
          f"max off-diagonal = {(ortho_check - torch.eye(K, device=ortho_check.device)).abs().max().item():.6f}")

    # Save PCA model
    pca_model = {
        'mu': mu.cpu(),        # (d, 1)
        'U': U_basis.cpu(),    # (d, K)
        'S': S.cpu(),          # (K,)
        'K': K,
        'N_templates': N,
        'N_discarded': discarded,
        'H': IMG_H,
        'W': IMG_W,
    }

    torch.save(pca_model, str(PCA_MODEL_PATH))
    print(f"\nPCA model saved to: {PCA_MODEL_PATH}")
    print(f"  μ shape: {mu.shape}")
    print(f"  U shape: {U_basis.shape}")
    print(f"  Top-5 singular values: {S[:5].tolist()}")

    return pca_model


# ═══════════════════════════════════════════════════════════════════════
#  PASS 2: Apply global PCA to reconstruct B_t and R_t for every frame
# ═══════════════════════════════════════════════════════════════════════

def _process_and_save_batch(batch_tensors, batch_paths, mu_gpu, U_gpu, device):
    """Process a batch of frames through PCA in one GPU shot."""
    # Stack all frames into one matrix and push to GPU ONCE
    # Shape: (d, BATCH_SIZE)
    X_batch = torch.cat(batch_tensors, dim=1).to(device)

    # 100% GPU Utilization Matrix Math
    X_centered = X_batch - mu_gpu
    coeffs = torch.mm(U_gpu.t(), X_centered)      # (K, d) @ (d, B) = (K, B)
    X_hat = torch.mm(U_gpu, coeffs) + mu_gpu       # (d, K) @ (K, B) = (d, B)

    # Calculate Residual
    R_batch = X_batch - X_hat

    # Pull entire batch back to CPU ONCE
    I_np = X_batch.cpu().numpy()
    B_np = X_hat.cpu().numpy()
    R_np = R_batch.cpu().numpy()

    # Save to disk
    for i, img_path in enumerate(batch_paths):
        # Extract the i-th column and reshape to 288x384
        i_t = I_np[:, i].reshape(IMG_H, IMG_W).clip(0, 255).astype(np.uint8)
        b_t = B_np[:, i].reshape(IMG_H, IMG_W).clip(0, 255).astype(np.uint8)
        r_raw = np.abs(R_np[:, i]).reshape(IMG_H, IMG_W)
        r_min, r_max = r_raw.min(), r_raw.max()
        if r_max - r_min > 1e-6:
            r_t = ((r_raw - r_min) / (r_max - r_min) * 255).astype(np.uint8)
        else:
            r_t = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        X_t = cv2.merge([r_t, b_t, i_t])

        rel_path = os.path.relpath(img_path, str(IMAGES_DIR))
        out_path = os.path.join(str(OUTPUT_DIR), rel_path)
        out_path = os.path.splitext(out_path)[0] + '.png'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, X_t)


def apply_pca_worker(args):
    """
    Worker function: applies the pre-learned PCA model to a batch of frames.
    Uses tensor batching (256 frames at once) to fully saturate V100 GPU.

    For each batch of frames:
      X_batch = [vec(I_1), ..., vec(I_256)]   → (d, 256)
      B_batch = U @ U^T @ (X_batch - μ) + μ   → one massive GPU matmul
      R_batch = X_batch - B_batch
    """
    img_paths, device_id, mu, U = args

    cv2.setNumThreads(0)
    torch.set_num_threads(1)

    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')

    # Move PCA model to this GPU
    mu_gpu = mu.to(device)   # (d, 1)
    U_gpu = U.to(device)     # (d, K)

    BATCH_SIZE = 256  # Load 256 frames at once
    batch_tensors = []
    batch_paths = []

    for img_path in img_paths:
        # Skip already generated files
        rel_path = os.path.relpath(img_path, str(IMAGES_DIR))
        out_path = os.path.join(str(OUTPUT_DIR), rel_path)
        out_path = os.path.splitext(out_path)[0] + '.png'
        if os.path.exists(out_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        if h != IMG_H or w != IMG_W:
            img = cv2.resize(img, (IMG_W, IMG_H))

        # Keep on CPU until batch is full
        x_t = torch.from_numpy(img.astype(np.float32)).view(-1, 1)
        batch_tensors.append(x_t)
        batch_paths.append(img_path)

        # When batch is full, send to GPU and process!
        if len(batch_tensors) == BATCH_SIZE:
            _process_and_save_batch(batch_tensors, batch_paths, mu_gpu, U_gpu, device)
            batch_tensors = []
            batch_paths = []

    # Catch any leftover frames at the end
    if len(batch_tensors) > 0:
        _process_and_save_batch(batch_tensors, batch_paths, mu_gpu, U_gpu, device)


def pass2_apply_pca(split='train'):
    """
    Pass 2: Load the global PCA model and apply it to every frame,
    generating the 3-channel [I_t, B_t, R_t] tensors.
    """
    print("\n" + "=" * 70)
    print("PASS 2: Applying Global PCA to Generate 3-Channel Tensors")
    print("=" * 70)

    # Load PCA model
    pca_model = torch.load(str(PCA_MODEL_PATH))
    mu = pca_model['mu']   # (d, 1)
    U = pca_model['U']     # (d, K)
    print(f"Loaded PCA model: K={pca_model['K']}, "
          f"trained on {pca_model['N_templates']} templates")

    # Gather all images
    images_dir = IMAGES_DIR / split
    all_images = sorted(
        glob.glob(str(images_dir / "*.jpg")) +
        glob.glob(str(images_dir / "*.png"))
    )
    print(f"Total frames to process: {len(all_images)}")

    # Distribute across GPUs 2-7
    gpu_list = [2, 3, 4, 5, 6, 7]
    num_workers = 36  # 6 workers per GPU

    # Split images into chunks for each worker
    chunk_size = (len(all_images) + num_workers - 1) // num_workers
    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(all_images))
        if start >= len(all_images):
            break
        gpu_id = gpu_list[i % len(gpu_list)]
        tasks.append((all_images[start:end], gpu_id, mu, U))

    print(f"Spawning {len(tasks)} workers across GPUs {gpu_list}...")

    # Use imap_unordered with a progress wrapper
    # Since each worker processes a batch, we track worker completions
    with mp.Pool(processes=len(tasks)) as pool:
        list(tqdm(
            pool.imap_unordered(apply_pca_worker, tasks),
            total=len(tasks),
            desc="Pass 2: PCA Apply (workers)"
        ))

    print("\n3-Channel Tensor Generation Complete!")
    print(f"Output saved to: {OUTPUT_DIR / split}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    mp.set_start_method('spawn', force=True)

    # Pass 1: Build templates and learn global PCA
    if PCA_MODEL_PATH.exists():
        print(f"PCA model already exists at {PCA_MODEL_PATH}")
        response = input("Re-run Pass 1? (y/N): ").strip().lower()
        if response == 'y':
            pass1_build_pca_model(split='train')
    else:
        pass1_build_pca_model(split='train')

    # Pass 2: Apply to all frames
    # pass2_apply_pca(split='train')
    pass2_apply_pca(split='val')


if __name__ == '__main__':
    main()
