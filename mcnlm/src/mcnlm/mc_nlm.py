# Montecarlo Non-Local Means

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit, prange

from mcnlm.utils import load_image, add_gaussian_noise

# Params
@dataclass
class MCNLMParams:
    sigma: float
    h_factor: float = 0.4
    patch_size: int = 5
    search_radius: int = 10
    spatial_sigma: float = 10.0
    sampling_prob: float = 0.5

    @property
    def patch_radius(self):
        return self.patch_size // 2

# MCNLM Kernel
def mcnlm_local(y_patch, window_patches, window_coords, r, f, sigma, h, sampling_prob):
    n = len(window_patches)

    mask = np.random.rand(n) < sampling_prob
    if not np.any(mask):
        return y_patch[len(y_patch) // 2]

    patches = window_patches[mask]
    # coords = window_coords[mask]

    h2 = h * h
    sigma2 = sigma * sigma

    diffs = patches - y_patch
    d2 = np.mean(diffs * diffs, axis=1)
    d2 = np.maximum(d2 - 2.0 * sigma2, 0.0)
    w_r = np.exp(-d2 / h2)

    # spatial_d2 = np.sum(coords * coords, axis=1)
    # w_s = np.exp(-spatial_d2 / (2.0 * params.spatial_sigma**2))

    w = w_r

    s = np.sum(w)
    if s == 0.0:
        return y_patch[len(y_patch) // 2]

    center = patches[:, len(y_patch) // 2]

    return np.sum(w * center) / s


# Sliding window utils
def extract_search_window(padded, pi, pj, rho, pad):
    r0, r1 = pi - rho - pad, pi + rho + pad + 1
    c0, c1 = pj - rho - pad, pj + rho + pad + 1
    return padded[r0:r1, c0:c1]


def window_patches(window, patch_size):
    patches = sliding_window_view(window, (patch_size, patch_size))
    return patches.reshape(-1, patch_size * patch_size)


def window_coords(rho):
    y, x = np.mgrid[-rho : rho + 1, -rho : rho + 1]
    return np.stack((y.flatten(), x.flatten()), axis=1)


# MCNLM
# @njit(parallel=True)
def mcnlm_denoise(noisy, r, f, sigma, h, sampling_prob):
    out = np.zeros_like(noisy)

    patch_radius = f // 2
    total_pad = r + patch_radius

    coords = window_coords(r)

    for i in range(noisy.shape[0] - 2 * total_pad):
        for j in range(noisy.shape[1] - 2 * total_pad):
            pi, pj = i + total_pad, j + total_pad

            y_patch = noisy[pi - patch_radius : pi + patch_radius + 1, 
                            pj - patch_radius : pj + patch_radius + 1].flatten()

            window = extract_search_window(noisy, pi, pj, r, patch_radius)
            patches = window_patches(window, f)

            out[pi, pj] = mcnlm_local(y_patch, patches, coords, r, f, sigma, h, sampling_prob)

    print()
    return out


def test_mcnlm(noisy, params: MCNLMParams):
    total_pad = params.search_radius + params.patch_radius
    padded = np.pad(noisy, total_pad, mode="reflect")
    denoised = mcnlm_denoise(padded, 
                             params.search_radius, 
                             params.patch_size, 
                             params.sigma, 
                             params.h_factor * params.sigma, 
                             params.sampling_prob)

    # Remove padding
    denoised = denoised[total_pad:-total_pad, total_pad:-total_pad]

    return denoised



def show_matches(image_path, points, K=10000):
    """
    Show strong non-local matches for a list of points.
    """
    image = load_image(image_path)

    SIGMA = 17
    noisy = add_gaussian_noise(image * 255, sigma=SIGMA).astype(np.float32) / 255.0
    image = noisy

    params = MCNLMParams(
        sigma = SIGMA / 255.0,
        h_factor = 0.4,
        patch_size = 5,
        search_radius = 20,
        spatial_sigma = 10,
        sampling_prob = 0.3
    )

    pad = params.patch_radius
    rho = params.search_radius
    total_pad = pad + rho
    padded = np.pad(image, total_pad, mode="reflect")
    coords = window_coords(rho)

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")

    for pi, pj in points:
        pi0, pj0 = pi + total_pad, pj + total_pad

        y_patch = padded[
            pi0 - pad : pi0 + pad + 1,
            pj0 - pad : pj0 + pad + 1
        ].flatten()

        window = extract_search_window(padded, pi0, pj0, rho, pad)
        patches = window_patches(window, params.patch_size)

        # --- compute full NLM weights ---
        diffs = patches - y_patch
        d2 = np.mean(diffs * diffs, axis=1)
        d2 = np.maximum(d2 - 2 * params.sigma**2, 0)

        h2 = (params.h_factor * params.sigma * params.patch_size) ** 2
        w_r = np.exp(-d2 / h2)

        spatial_d2 = np.sum(coords * coords, axis=1)
        w_s = np.exp(-spatial_d2 / (2 * params.spatial_sigma**2))

        w = w_r * w_s

        # keep strongest matches
        idx = np.argsort(w)[-K:]
        ys = pi + coords[idx,0]
        xs = pj + coords[idx,1]

        plt.scatter(xs, ys, c=w[idx], s=10, cmap="hot")
        plt.scatter([pj], [pi], c="green", s=10)

    plt.title("Strong non-local matches for multiple points")
    plt.axis("off")
    plt.show()
