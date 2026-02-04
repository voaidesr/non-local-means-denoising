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
    seed: int | None = None,
    show: bool = True,
):
    """
    Zoomed standard comparison for hashed NLM.
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
