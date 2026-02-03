from mcnlm.utils import mse, add_gaussian_noise, load_image, psnr, estimate_noise
import numpy as np
import time
from scipy.spatial import KDTree
from mcnlm.utils import save_image


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


def run_kdtree_naive(image: np.ndarray, patch_size: int):
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

    denoised = np.zeros_like(image)

    for idx in range(n_patches):
        patch = patches[idx, :]
        coord = coords[idx, :]

        time_query_start = time.time()
        distances, indices = kdtree.query(patch, k=10)
        time_query_end = time.time()
        if idx % 1000 == 0:
            print(f"Patch {idx}/{n_patches}: K-D tree query took {time_query_end - time_query_start:.4f} seconds")

        similar_patches = patches[indices, :]
        denoised_patch = np.mean(similar_patches, axis=0)

        i, j = coord
        denoised[i:i+patch_size, j:j+patch_size] += denoised_patch.reshape((patch_size, patch_size))

    denoised /= 10  # Average overlapping patches

    return denoised




def kdtree_nlm(image_path: str, output_path: str) -> None:
    image = load_image(image_path)
    sigma = 17.0
    noisy = add_gaussian_noise(image * 255, sigma) / 255.0
    
    patch_size = 5
    padded_noisy = np.pad(noisy, patch_size // 2, mode='reflect')

    # Save denoised image
    denoised = run_kdtree_naive(padded_noisy, patch_size)
    import matplotlib.pyplot as plt
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image using K-D Tree NLM')
    plt.axis('off')
    plt.show()

    # save_image(denoised, output_path)

    print(f"Denoised image saved to {output_path}")