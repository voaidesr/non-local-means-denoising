import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'src')

from mcnlm.utils import load_image, add_gaussian_noise
from mcnlm.kdtree_nlm import extract_patches as kdtree_nlm_extract_patches
from scipy.spatial import KDTree

def analyze_patch_selection(test_pixels_offsets: list[tuple[int, int]] = None):
    original = load_image("imgs/clock.tiff")
    sigma = 17.0
    noisy = add_gaussian_noise(original.copy() * 255, sigma).astype(np.float32) / 255.0

    patch_size = 5
    search_radius = 10
    k_neighbors = 100

    patches, coords = kdtree_nlm_extract_patches(noisy, patch_size)

    tree = KDTree(patches)

    _, axes = plt.subplots(1, len(test_pixels_offsets), figsize=(20, 8))
    for idx, (test_i, test_j) in enumerate(test_pixels_offsets):
        test_i += noisy.shape[0] // 2
        test_j += noisy.shape[1] // 2

        test_idx = test_i * noisy.shape[1] + test_j
        _, indices = tree.query(patches[test_idx:test_idx+1], k=k_neighbors)
        knn_indices = indices[0]
        knn_coords = coords[knn_indices]

        ax = axes[idx] if len(test_pixels_offsets) > 1 else axes
        ax.imshow(noisy, cmap='gray')
        ax.scatter(knn_coords[:, 1], knn_coords[:, 0], c='red', s=5, alpha=0.5, label='k-NN')
        ax.scatter([test_j], [test_i], c='lime', s=100, marker='x', linewidths=3, label='Test pixel')
        # Draw local search window
        rect = plt.Rectangle((test_j - search_radius, test_i - search_radius),
                            2*search_radius+1, 2*search_radius+1,
                            fill=False, edgecolor='blue', linewidth=2, label='Local window')
        ax.add_patch(rect)
        ax.set_title(f'k-NN locations vs within local window')
        ax.legend(loc='upper right')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('../docs/res/knn_vs_mc_spatial.pdf', format='pdf')

if __name__ == "__main__":
    analyze_patch_selection(([0, 0], [60, -40], [0, 40]))