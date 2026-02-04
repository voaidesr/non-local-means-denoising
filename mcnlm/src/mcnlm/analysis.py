import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mcnlm.utils import load_image, add_gaussian_noise
from mcnlm.kdtree import (
    extract_patches,
    patch_centers,
    center_to_patch_index,
    local_window_indices,
    patch_distances,
    kdtree_knn,
)


def analyze_patch_selection(
    image_path: str,
    output_path: str,
    test_pixels_offsets: list[tuple[int, int]],
    patch_size: int = 5,
    search_radius: int = 10,
    k_neighbors: int = 100,
    sigma: float = 17.0,
    pca_components: int | None = None,
    seed: int | None = None,
    show: bool = False,
):
    original = load_image(image_path)
    if seed is not None:
        np.random.seed(seed)
    noisy = add_gaussian_noise(original.copy() * 255, sigma).astype(np.float32) / 255.0

    patches, coords = extract_patches(noisy, patch_size)
    height, width = noisy.shape
    coords_center = patch_centers(coords, patch_size)

    test_indices = []
    centers = []
    for test_i, test_j in test_pixels_offsets:
        center_i = height // 2 + test_i
        center_j = width // 2 + test_j
        test_idx = center_to_patch_index(center_i, center_j, height, width, patch_size)
        test_indices.append(test_idx)
        centers.append((center_i, center_j))

    knn_indices_all, _ = kdtree_knn(patches, k_neighbors=k_neighbors, n_components=pca_components)

    _, axes = plt.subplots(1, len(test_pixels_offsets), figsize=(20, 8))
    for idx, (center_i, center_j) in enumerate(centers):
        test_idx = test_indices[idx]
        knn_indices = knn_indices_all[test_idx]
        knn_coords = coords_center[knn_indices]

        ax = axes[idx] if len(test_pixels_offsets) > 1 else axes
        ax.imshow(noisy, cmap='gray')
        ax.scatter(knn_coords[:, 1], knn_coords[:, 0], c='red', s=5, alpha=0.5, label='k-NN')
        ax.scatter([center_j], [center_i], c='lime', s=100, marker='x', linewidths=3, label='Test pixel')

        rect = plt.Rectangle((center_j - search_radius, center_i - search_radius),
                             2 * search_radius + 1, 2 * search_radius + 1,
                             fill=False, edgecolor='blue', linewidth=2, label='Local window')
        ax.add_patch(rect)
        ax.set_title('k-NN locations vs within local window')
        ax.legend(loc='upper right')
        ax.axis('off')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), format='pdf')
    if show:
        plt.show()
    else:
        plt.close()


def analyze_kdtree_spatial(
    image_path: str,
    output_path: str,
    test_pixel_offset: tuple[int, int] = (0, 0),
    patch_size: int = 5,
    search_radius: int = 10,
    k_neighbors: int = 100,
    sigma: float = 17.0,
    pca_components: int | None = None,
    seed: int | None = None,
    show: bool = False,
    sample_size: int = 5000,
):
    """Detailed k-NN vs local window spatial analysis with diagnostic plots."""
    original = load_image(image_path)
    if seed is not None:
        np.random.seed(seed)
    noisy = add_gaussian_noise(original.copy() * 255, sigma).astype(np.float32) / 255.0
    noisy = np.clip(noisy, 0.0, 1.0)

    height, width = noisy.shape

    patches, coords = extract_patches(noisy, patch_size)
    coords_center = patch_centers(coords, patch_size)

    center_i = height // 2 + test_pixel_offset[0]
    center_j = width // 2 + test_pixel_offset[1]
    test_idx = center_to_patch_index(center_i, center_j, height, width, patch_size)

    knn_indices_all, knn_distances_sq_all = kdtree_knn(
        patches,
        k_neighbors=k_neighbors,
        n_components=pca_components,
    )
    knn_indices = knn_indices_all[test_idx]
    knn_distances_sq = knn_distances_sq_all[test_idx]
    knn_coords = coords_center[knn_indices]

    spatial_distances = np.sqrt((knn_coords[:, 0] - center_i) ** 2 + (knn_coords[:, 1] - center_j) ** 2)
    within_window = np.sum(
        (np.abs(knn_coords[:, 0] - center_i) <= search_radius) &
        (np.abs(knn_coords[:, 1] - center_j) <= search_radius)
    )

    local_indices = local_window_indices(coords_center, center_i, center_j, search_radius)
    local_distances_sq = patch_distances(patches, test_idx, local_indices)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(noisy, cmap='gray')
    axes[0].scatter(knn_coords[:, 1], knn_coords[:, 0], c='red', s=5, alpha=0.5, label='k-NN')
    axes[0].scatter([center_j], [center_i], c='lime', s=100, marker='x', linewidths=3, label='Test pixel')
    rect = plt.Rectangle((center_j - search_radius, center_i - search_radius),
                         2 * search_radius + 1, 2 * search_radius + 1,
                         fill=False, edgecolor='blue', linewidth=2, label='Local window')
    axes[0].add_patch(rect)
    axes[0].set_title(f'k-NN locations (k={k_neighbors})\n{within_window} within local window')
    axes[0].legend(loc='upper right')
    axes[0].axis('off')

    axes[1].hist(spatial_distances, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=search_radius, color='red', linestyle='--', linewidth=2, label=f'search_radius={search_radius}')
    axes[1].set_xlabel('Spatial distance from test pixel')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Spatial distribution of k-NN')
    axes[1].legend()

    all_spatial_dist = np.sqrt((coords_center[:, 0] - center_i) ** 2 + (coords_center[:, 1] - center_j) ** 2)
    all_patch_dist_sq = patch_distances(patches, test_idx, np.arange(len(patches)))
    all_patch_dist = np.sqrt(all_patch_dist_sq)
    sample_n = min(sample_size, len(patches))
    sample_idx = np.random.choice(len(patches), sample_n, replace=False)

    axes[2].scatter(all_spatial_dist[sample_idx], all_patch_dist[sample_idx],
                    c='gray', s=1, alpha=0.3, label='All patches')
    axes[2].scatter(spatial_distances, np.sqrt(knn_distances_sq), c='red', s=20, alpha=0.7, label='k-NN')
    axes[2].axvline(x=search_radius, color='blue', linestyle='--', linewidth=2, label=f'search_radius={search_radius}')
    axes[2].set_xlabel('Spatial distance')
    axes[2].set_ylabel('Patch distance (Euclidean)')
    axes[2].set_title('Patch similarity vs spatial distance')
    axes[2].legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    sigma_normalized = sigma / 255.0
    h = 0.4 * sigma_normalized
    h2 = h * h
    sigma2 = sigma_normalized * sigma_normalized

    knn_d2_normalized = knn_distances_sq / patches.shape[1]
    knn_d2_adjusted = np.maximum(knn_d2_normalized - 2 * sigma2, 0.0)
    knn_weights = np.exp(-knn_d2_adjusted / h2)

    local_d2_normalized = local_distances_sq / patches.shape[1]
    local_d2_adjusted = np.maximum(local_d2_normalized - 2 * sigma2, 0.0)
    local_weights = np.exp(-local_d2_adjusted / h2)

    print(f"Image shape: {noisy.shape}")
    print(f"Number of patches: {len(patches)}")
    print(f"Patch dimension: {patches.shape[1]}")
    print(f"Test pixel: ({center_i}, {center_j}), patch index: {test_idx}")
    print(f"K-D Tree k-NN analysis (k={k_neighbors}):")
    print(f"  Min spatial distance: {spatial_distances.min():.2f}")
    print(f"  Max spatial distance: {spatial_distances.max():.2f}")
    print(f"  Mean spatial distance: {spatial_distances.mean():.2f}")
    print(f"  Median spatial distance: {np.median(spatial_distances):.2f}")
    print(f"  Neighbors within search_radius={search_radius}: {within_window}/{k_neighbors} ({100 * within_window / k_neighbors:.1f}%)")
    print(f"Local window patches: {len(local_indices)}")
    print(f"  Min patch distance in local window: {np.sqrt(local_distances_sq).min():.4f}")
    print(f"  Max patch distance in local window: {np.sqrt(local_distances_sq).max():.4f}")
    print(f"  Mean patch distance in local window: {np.sqrt(local_distances_sq).mean():.4f}")
    print(f"k-NN patches in local window: {np.sum(np.isin(knn_indices, local_indices))}/{k_neighbors}")
    print("NLM Weight Analysis")
    print(f"  k-NN total weight: {knn_weights.sum():.4f}")
    print(f"  Local window total weight: {local_weights.sum():.4f}")
