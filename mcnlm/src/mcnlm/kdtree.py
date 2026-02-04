from mcnlm.utils import mse, add_gaussian_noise, load_image, psnr, estimate_noise
import numpy as np
import time
from scipy.spatial import KDTree
from mcnlm.utils import save_image
from numba import njit, prange
from sklearn.decomposition import PCA


def extract_patches(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    patches = np.zeros(( (height - patch_size + 1) * (width - patch_size + 1), patch_size * patch_size))
    coords = np.zeros(((height - patch_size + 1) * (width - patch_size + 1), 2), dtype=int)
    idx = 0

    for i in range(height - patch_size + 1):
        for j in range(width - patch_size + 1):
            patch = image[i:i+patch_size, j:j+patch_size].flatten()
            patches[idx, :] = patch
            coords[idx, :] = (i, j)
            idx += 1

    return patches, coords


@njit(parallel=True)
def aggregate_denoised_patches_nlm(denoised: np.ndarray, weights_sum: np.ndarray,
                                    patches: np.ndarray, distances: np.ndarray, 
                                    indices: np.ndarray, coords: np.ndarray, 
                                    patch_size: int, n_patches: int, h_squared: float,
                                    sigma_squared: float):
    half = patch_size // 2
    center_idx = half * patch_size + half
    patch_dim = patch_size * patch_size
    
    for idx in prange(n_patches):
        neighbor_indices = indices[idx, :]
        neighbor_distances = distances[idx, :]
        
        i, j = coords[idx, 0], coords[idx, 1]
        center_i = i + half
        center_j = j + half
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for ki in range(len(neighbor_indices)):
            ni = neighbor_indices[ki]
            
            dist_sq = neighbor_distances[ki] / patch_dim
            dist_sq = max(dist_sq - 2.0 * sigma_squared, 0.0)
            weight = np.exp(-dist_sq / h_squared)
            center_value = patches[ni, center_idx]
            
            weighted_sum += weight * center_value
            weight_total += weight
        
        if weight_total > 0:
            denoised[center_i, center_j] += weighted_sum
            weights_sum[center_i, center_j] += weight_total


@njit(parallel=True)
def recompute_distances(patches: np.ndarray, indices: np.ndarray, patch_dim: int) -> np.ndarray:
    n_patches = indices.shape[0]
    k = indices.shape[1]
    distances = np.zeros((n_patches, k), dtype=np.float64)
    
    for idx in prange(n_patches):
        patch_i = patches[idx, :]
        for ki in range(k):
            ni = indices[idx, ki]
            patch_j = patches[ni, :]
            dist_sq = 0.0
            for d in range(patch_dim):
                diff = patch_i[d] - patch_j[d]
                dist_sq += diff * diff
            distances[idx, ki] = dist_sq
    
    return distances


def run_kdtree_naive(image: np.ndarray, patch_size: int, k_neighbors: int = 1000, 
                      sigma: float = 0.0, h_factor: float = 0.55):
    time_extract = time.time()
    patches, coords = extract_patches(image, patch_size)
    time_extract_end = time.time()
    print(f"Patch extraction took {time_extract_end - time_extract:.2f} seconds")

    n_patches = patches.shape[0]
    patch_dim = patches.shape[1]
    assert patch_dim == patch_size * patch_size, "Patch dimension mismatch"
    print(f"Extracted {n_patches} patches of size {patch_size}x{patch_size} (dim={patch_dim})")

    pca_start = time.time()
    n_components = min(10, patch_dim)
    pca = PCA(n_components=n_components)
    patches_reduced = pca.fit_transform(patches)
    pca_end = time.time()
    print(f"PCA reduction ({patch_dim}D -> {n_components}D) took {pca_end - pca_start:.2f} seconds")

    tree_build_start = time.time()
    kdtree = KDTree(patches_reduced)
    tree_build_end = time.time()
    print(f"KDTree build took {tree_build_end - tree_build_start:.2f} seconds")

    query_start = time.time()
    _, indices = kdtree.query(patches_reduced, k=k_neighbors)
    query_end = time.time()
    print(f"Batch KDTree query took {query_end - query_start:.2f} seconds")
    
    dist_start = time.time()
    distances = recompute_distances(patches, indices, patch_dim)
    dist_end = time.time()
    print(f"Distance recomputation took {dist_end - dist_start:.2f} seconds")
    
    h = h_factor * sigma
    h_squared = h * h
    sigma_squared = sigma * sigma

    denoised = np.zeros_like(image)
    weights_sum = np.zeros_like(image)

    aggregate_start = time.time()
    aggregate_denoised_patches_nlm(denoised, weights_sum, patches, distances, 
                                    indices, coords, patch_size, n_patches, 
                                    h_squared, sigma_squared)
    aggregate_end = time.time()
    print(f"Parallel aggregation took {aggregate_end - aggregate_start:.2f} seconds")

    # Normalize by weight sums
    denoised = np.divide(denoised, weights_sum, where=weights_sum > 0)

    return denoised


def kdtree_nlm(image_path: str, output_path: str) -> None:
    image = load_image(image_path)
    sigma = 17.0
    noisy = add_gaussian_noise(image * 255, sigma) / 255.0

    patch_size = 5
    padded_noisy = np.pad(noisy, patch_size // 2, mode='reflect')

    denoised = run_kdtree_naive(padded_noisy, patch_size, sigma=sigma/255.0)

    pad = patch_size // 2
    denoised = denoised[pad:-pad, pad:-pad]

    noisy *= 255.0
    denoised *= 255.0

    mse_noisy = mse(image.copy(), noisy)
    psnr_noisy = psnr(image.copy(), noisy)
    mse_denoised = mse(image.copy(), denoised)
    psnr_denoised = psnr(image.copy(), denoised)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image (MSE={:.4f}, PSNR={:.2f} dB)'.format(mse_noisy, psnr_noisy))
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image (MSE={:.4f}, PSNR={:.2f} dB)'.format(mse_denoised, psnr_denoised))
    plt.axis('off')
    plt.show()

    print(f"Denoised image saved to {output_path}")