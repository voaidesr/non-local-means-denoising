import numpy as np
import matplotlib.pyplot as plt
import cv2

from mcnlm.hashnlm import denoise_hashnlm


def standard_comparison_hashed_nlm(
    img_path,
    output_path,
    sigma=17,
    num_features=4,
    beta=0.88,
    seed: int | None = None,
    show: bool = True,
):
    """
    Standard comparison for hashed NLM.
    Please don't modify normalization of images, because they go crazy.
    """
    # some functions required for this comparison
    # functions from utils mess up everything, numpy fails miserably
    def load_image(path, fallback_size=(100, 100)):
        img = cv2.imread(path, 0)
        if img is None:
            img = np.zeros(fallback_size)
            cv2.circle(img, (50, 50), 30, 1, -1)
        return img.astype(np.float64) / 255.0

    def add_noise(normalized_img, sigma_val):
        noise = np.random.normal(0, sigma_val, normalized_img.shape)
        noisy_img = normalized_img + noise
        return np.clip(noisy_img, 0, 1)

    def mse(img_a, img_b):
        err = np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)
        err /= float(img_a.shape[0] * img_a.shape[1])
        return err

    def psnr(img_a, img_b, max_pixel=1.0):
        mse_val = mse(img_a, img_b)
        if mse_val == 0:
            return float("inf")

        return 10 * np.log10((max_pixel ** 2) / mse_val)

    sigma /= 255.0

    if seed is not None:
        np.random.seed(seed)

    original_image = load_image(img_path)
    noisy_image = add_noise(original_image.copy(), sigma)

    denoised_image = denoise_hashnlm(
        noisy_image.copy(), sigma, num_features, beta)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original (Clean)")
    plt.subplot(132)
    plt.imshow(noisy_image, cmap="gray")
    plt.title(
        f"Noisy MSE = {mse(original_image.copy()*255.0, noisy_image.copy()*255.0):.2f} | "
        f"PSNR = {psnr(original_image, noisy_image):.4f}"
    )
    plt.subplot(133)
    plt.imshow(denoised_image, cmap="gray")
    plt.title(
        f"Denoised MSE = {mse(original_image.copy()*255.0, denoised_image.copy()*255.0):.2f} | "
        f"PSNR = {psnr(original_image, denoised_image):.4f}"
    )

    plt.suptitle(
        f"Hashed NLM sigma={sigma:.3f}, beta={beta}, #features={num_features}")

    plt.savefig(output_path)
    if show:
        plt.show()
    else:
        plt.close()


def standard_comparison_hashed_nlm_zoomed(
    img_path,
    output_path,
    sigma=17,
    num_features=4,
    beta=0.88,
    zoom_size=64,
    zoom_center=None,
    seed: int | None = None,
    show: bool = True,
):
    """
    Zoomed standard comparison for hashed NLM.
    Please don't modify normalization of images, because they go crazy.

    zoom_size is the side length (N) of the N x N zoom region.
    zoom_center defaults to the image center (x, y).
    """
    # some functions required for this comparison
    # functions from utils mess up everything, numpy fails miserably
    def load_image(path, fallback_size=(100, 100)):
        img = cv2.imread(path, 0)
        if img is None:
            img = np.zeros(fallback_size)
            cv2.circle(img, (50, 50), 30, 1, -1)
        return img.astype(np.float64) / 255.0

    def add_noise(normalized_img, sigma_val):
        noise = np.random.normal(0, sigma_val, normalized_img.shape)
        noisy_img = normalized_img + noise
        return np.clip(noisy_img, 0, 1)

    def mse(img_a, img_b):
        err = np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)
        err /= float(img_a.shape[0] * img_a.shape[1])
        return err

    def psnr(img_a, img_b, max_pixel=1.0):
        mse_val = mse(img_a, img_b)
        if mse_val == 0:
            return float("inf")

        return 10 * np.log10((max_pixel ** 2) / mse_val)

    sigma /= 255.0

    if seed is not None:
        np.random.seed(seed)

    original_image = load_image(img_path)
    noisy_image = add_noise(original_image.copy(), sigma)

    denoised_image = denoise_hashnlm(
        noisy_image.copy(), sigma, num_features, beta)

    height, width = original_image.shape
    zoom_size = int(zoom_size)
    zoom_size = max(1, min(zoom_size, height, width))

    if zoom_center is None:
        cx, cy = width // 2, height // 2
    else:
        cx, cy = zoom_center
    cx = int(cx)
    cy = int(cy)

    x0 = max(0, min(cx - zoom_size // 2, width - zoom_size))
    y0 = max(0, min(cy - zoom_size // 2, height - zoom_size))
    x1 = x0 + zoom_size
    y1 = y0 + zoom_size

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].imshow(original_image, cmap="gray")
    axs[0, 0].set_title("Original (Clean)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(noisy_image, cmap="gray")
    axs[0, 1].set_title(
        f"Noisy MSE = {mse(original_image.copy()*255.0, noisy_image.copy()*255.0):.2f} | "
        f"PSNR = {psnr(original_image, noisy_image):.4f}"
    )
    axs[0, 1].axis("off")

    axs[0, 2].imshow(denoised_image, cmap="gray")
    axs[0, 2].set_title(
        f"Denoised MSE = {mse(original_image.copy()*255.0, denoised_image.copy()*255.0):.2f} | "
        f"PSNR = {psnr(original_image, denoised_image):.4f}"
    )
    axs[0, 2].axis("off")

    rect = plt.Rectangle((x0, y0), zoom_size, zoom_size,
                         edgecolor="red", facecolor="none", linewidth=1, alpha=0.6)
    axs[0, 0].add_patch(rect)
    rect = plt.Rectangle((x0, y0), zoom_size, zoom_size,
                         edgecolor="red", facecolor="none", linewidth=1, alpha=0.6)
    axs[0, 1].add_patch(rect)
    rect = plt.Rectangle((x0, y0), zoom_size, zoom_size,
                         edgecolor="red", facecolor="none", linewidth=1, alpha=0.6)
    axs[0, 2].add_patch(rect)

    axs[1, 0].imshow(original_image[y0:y1, x0:x1], cmap="gray")
    axs[1, 0].set_title(f"Zoom ({zoom_size}x{zoom_size})")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(noisy_image[y0:y1, x0:x1], cmap="gray")
    axs[1, 1].set_title("Zoom")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(denoised_image[y0:y1, x0:x1], cmap="gray")
    axs[1, 2].set_title("Zoom")
    axs[1, 2].axis("off")

    fig.suptitle(
        f"Hashed NLM sigma={sigma:.3f}, beta={beta}, #features={num_features}")
    plt.tight_layout()

    plt.savefig(output_path)
    if show:
        plt.show()
    else:
        plt.close()
