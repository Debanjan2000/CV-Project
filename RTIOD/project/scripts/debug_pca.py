"""
Debug script: Run both passes on a small subset and save all 4 outputs
(original, background, residual, 3-channel) for visual verification.
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR.parent / "data" / "images" / "train"
OUTPUT_DIR = BASE_DIR.parent / "debug_pca"
PCA_MODEL_PATH = BASE_DIR.parent / "data" / "pca_model.pt"

IMG_H, IMG_W = 288, 384

TARGETS = [
    "frames_20200514_clip_7_1647_image_0057.jpg",
    "frames_20200514_clip_8_1715_image_0009.jpg",
    "frames_20200526_clip_31_1412_image_0059_dup2.jpg",
    "frames_20200627_clip_40_1802_image_0031.jpg",
    "frames_20200709_clip_36_1551_image_0026_dup1.jpg",
    "frames_20200812_clip_42_1834_image_0103_dup2.jpg",
    "frames_20200822_clip_47_2056_image_0040.jpg",
    "frames_20210201_clip_43_2100_image_0103.jpg"
]


def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # Load the global PCA model
    if not PCA_MODEL_PATH.exists():
        print(f"ERROR: PCA model not found at {PCA_MODEL_PATH}")
        print("Run 'python scripts/build_pca_dataset.py' first to learn the global PCA.")
        return

    pca_model = torch.load(str(PCA_MODEL_PATH), map_location=device)
    mu = pca_model['mu'].to(device)   # (d, 1)
    U = pca_model['U'].to(device)     # (d, K)
    K = pca_model['K']

    print(f"Loaded PCA model: K={K}, trained on {pca_model['N_templates']} templates")
    print(f"Processing {len(TARGETS)} debug targets...\n")

    for fname in TARGETS:
        img_path = str(IMAGES_DIR / fname)
        if not os.path.exists(img_path):
            print(f"  SKIP (not found): {fname}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  SKIP (unreadable): {fname}")
            continue

        h, w = img.shape
        if h != IMG_H or w != IMG_W:
            img = cv2.resize(img, (IMG_W, IMG_H))

        # Vectorize and project
        x_t = torch.from_numpy(img.astype(np.float32)).to(device).view(-1, 1)
        x_centered = x_t - mu
        coeffs = torch.mm(U.t(), x_centered)
        x_hat = torch.mm(U, coeffs) + mu

        B_t = x_hat.view(IMG_H, IMG_W)
        I_t = x_t.view(IMG_H, IMG_W)
        R_t = I_t - B_t

        # Convert to numpy
        i_np = I_t.cpu().numpy().clip(0, 255).astype(np.uint8)
        b_np = B_t.cpu().numpy().clip(0, 255).astype(np.uint8)
        r_raw = np.abs(R_t.cpu().numpy())
        r_min, r_max = r_raw.min(), r_raw.max()
        if r_max - r_min > 1e-6:
            r_np = ((r_raw - r_min) / (r_max - r_min) * 255).astype(np.uint8)
        else:
            r_np = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        x_np = cv2.merge([r_np, b_np, i_np])

        base = os.path.splitext(fname)[0]
        cv2.imwrite(str(OUTPUT_DIR / f"{base}_A_orig.png"), i_np)
        cv2.imwrite(str(OUTPUT_DIR / f"{base}_B_bg.png"), b_np)
        cv2.imwrite(str(OUTPUT_DIR / f"{base}_C_residual.png"), r_np)
        cv2.imwrite(str(OUTPUT_DIR / f"{base}_D_pca3channel.png"), x_np)
        print(f"  ✓ {fname}")

    print(f"\nDebug outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
