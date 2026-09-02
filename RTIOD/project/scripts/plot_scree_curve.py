import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_scree_curve():
    print("Sampling 1000 random thermal frames to compute PCA Eigenvalue Spectrum...")
    images_dir = "/raid/ai24mtech12009/cv_proj/RTIOD/startingkit/data/images/train/"
    
    # Grab all images and randomly sample 1000 to save memory and time
    all_images = glob.glob(os.path.join(images_dir, "*.jpg"))
    np.random.seed(42)
    sampled_images = np.random.choice(all_images, 1000, replace=False)
    
    IMG_H, IMG_W = 288, 384
    templates = []
    
    for img_path in tqdm(sampled_images, desc="Loading Images"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H))
        templates.append(img.flatten().astype(np.float32))
        
    X = torch.from_numpy(np.stack(templates, axis=0)).cuda() # Shape: (1000, 110592)
    
    # Center the data
    mu = X.mean(dim=0, keepdim=True)
    X_centered = X - mu
    
    # Compute PCA for the top 200 components to see the full curve
    print("Computing PCA (q=200)... this may take a moment.")
    U, S, V = torch.pca_lowrank(X_centered, q=200, center=False)
    
    # Singular Values (S) relate to Eigenvalues (Lambda) by: Lambda = S^2 / (N - 1)
    eigenvalues = (S ** 2) / (X.shape[0] - 1)
    eigenvalues = eigenvalues.cpu().numpy()
    
    # Calculate cumulative explained variance
    total_variance_approx = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance_approx
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot the cumulative variance line
    plt.plot(range(1, 201), cumulative_variance, color='b', linewidth=2, label='Cumulative Variance')
    
    # Highlight k=50
    variance_at_50 = cumulative_variance[49]
    plt.axvline(x=50, color='r', linestyle='--', label=f'k=50 (Variance: {variance_at_50*100:.1f}%)')
    plt.scatter(50, variance_at_50, color='r', s=100, zorder=5)
    
    plt.title("PCA Scree Plot: Cumulative Explained Variance", fontsize=16)
    plt.xlabel("Number of Principal Components (k)", fontsize=14)
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save the plot
    output_path = "/raid/ai24mtech12009/cv_proj/RTIOD/startingkit/scree_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS! Scree plot saved to {output_path}")
    print(f"At k=50, the model captures {variance_at_50*100:.2f}% of the variance.")

if __name__ == "__main__":
    plot_scree_curve()
