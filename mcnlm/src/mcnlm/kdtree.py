from mcnlm.utils import mse, add_gaussian_noise, load_image, psnr, estimate_noise
import numpy as np
import time
from scipy.spatial import KDTree
from mcnlm.utils import save_image
from numba import njit, prange


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
def aggregate_denoised_patches(denoised: np.ndarray, counts: np.ndarray,
                                similar_patches: np.ndarray, indices: np.ndarray,
                                coords: np.ndarray, patch_size: int, n_patches: int):

    for idx in prange(n_patches):
        neighbor_indices = indices[idx, :]

        k = len(neighbor_indices)
        patch_dim = patch_size * patch_size
        denoised_patch = np.zeros(patch_dim)

        for ki in range(k):
            ni = neighbor_indices[ki]
            for pi in range(patch_dim):
                denoised_patch[pi] += similar_patches[ni, pi]

        for pi in range(patch_dim):
            denoised_patch[pi] /= k

        i, j = coords[idx, 0], coords[idx, 1]

        for di in range(patch_size):
            for dj in range(patch_size):
                pi = di * patch_size + dj
                denoised[i + di, j + dj] += denoised_patch[pi]
                counts[i + di, j + dj] += 1


def run_kdtree_naive(image: np.ndarray, patch_size: int, k_neighbors: int = 100):
    time_extract = time.time()
    patches, coords = extract_patches(image, patch_size)
    time_extract_end = time.time()
    print(f"Patch extraction took {time_extract_end - time_extract:.2f} seconds")

    n_patches = patches.shape[0]
    patch_dim = patches.shape[1]
    assert patch_dim == patch_size * patch_size, "Patch dimension mismatch"
    print(f"Extracted {n_patches} patches of size {patch_size}x{patch_size} (dim={patch_dim})")

    tree_build_start = time.time()
    kdtree = KDTree(patches)
    tree_build_end = time.time()
    print(f"K-D tree build took {tree_build_end - tree_build_start:.2f} seconds")

    # Batch query all patches at once
    query_start = time.time()
    _, indices = kdtree.query(patches, k=k_neighbors)
    query_end = time.time()
    print(f"Batch K-D tree query took {query_end - query_start:.2f} seconds")

    denoised = np.zeros_like(image)
    counts = np.zeros_like(image)

    aggregate_start = time.time()
    aggregate_denoised_patches(denoised, counts, patches, indices, coords, patch_size, n_patches)
    aggregate_end = time.time()
    print(f"Parallel aggregation took {aggregate_end - aggregate_start:.2f} seconds")

    # Normalize by counts
    denoised = np.divide(denoised, counts, where=counts > 0)

    return denoised


def kdtree_nlm(image_path: str, output_path: str) -> None:
    image = load_image(image_path)
    sigma = 17.0
    noisy = add_gaussian_noise(image * 255, sigma) / 255.0

    patch_size = 5
    padded_noisy = np.pad(noisy, patch_size // 2, mode='reflect')

    denoised = run_kdtree_naive(padded_noisy, patch_size)

    pad = patch_size // 2
    denoised = denoised[pad:-pad, pad:-pad]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image using K-D Tree NLM')
    plt.axis('off')
    plt.show()

    print(f"Denoised image saved to {output_path}")