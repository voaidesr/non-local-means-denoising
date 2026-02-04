import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm
from mcnlm.mc_convergence import mc_convergence, compare_noise_estimation

from mcnlm.utils import show_mcnlm_result_zoomed, show_matches, show_nlm_result_zoomed
import numpy as np
import sys


def results_mcnlm():
    show_mcnlm_result_zoomed(
        "imgs/land.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/mcnlm3.pdf",
    )

    show_mcnlm_result_zoomed(
        "imgs/clock.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/mcnlm1.pdf",
    )

    show_mcnlm_result_zoomed(
        "imgs/man.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(440, 600, 64, 64),
        output_path="../docs/res/mcnlm2.pdf",
    )

    show_matches('imgs/clock.tiff', [(150, 210), (90, 135),
                 (170, 80)], "../docs/res/strong_matches_p1.pdf")


def results_naive_nlm():
    show_nlm_result_zoomed(
        "imgs/land.tiff",
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/nlm_denoise1.pdf",
    )

    show_nlm_result_zoomed(
        "imgs/clock.tiff",
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/nlm_denoise2.pdf",
    )


def mc_convergence_results():
    mc_convergence(image_path='imgs/moon.tiff', output_path1='../docs/res/convergence1_mse.pdf',
                   output_path2='../docs/res/convergence1_psnr.pdf')
    mc_convergence(image_path='imgs/clock.tiff', output_path1='../docs/res/convergence2_mse.pdf',
                   output_path2='../docs/res/convergence2_psnr.pdf')


def noise_comparison_results():
    compare_noise_estimation(
        image_path='imgs/moon.tiff',
        output_path_visual='../docs/res/noise_comparison_visual.pdf',
        output_path_mse='../docs/res/noise_comparison_mse.pdf',
        output_path_psnr='../docs/res/noise_comparison_psnr.pdf'
    )

    compare_noise_estimation(
        image_path='imgs/clock.tiff',
        output_path_visual='../docs/res/noise_comparison_visual2.pdf',
        output_path_mse='../docs/res/noise_comparison_mse2.pdf',
        output_path_psnr='../docs/res/noise_comparison_psnr2.pdf'
    )


# --------- New functions I added for Hashed NLM visualisation -------------
def standard_comparison_hashed_nlm(img_path, output_path, sigma=17, num_features=4, beta=0.88):
    """
        This is the standard comparison for hashed NLM.
        Please don't modify normalization of images, because they go crazy
    """
    from mcnlm.hashnlm import denoise_hashnlm
    import matplotlib.pyplot as plt
    import cv2

    # some functions required for this comparison
    # functions from utils mess up everything, numpy fails miserably
    def load_image(path, fallback_size=(100, 100)):
        """
        Load an image from the specified path.
        """
        img = cv2.imread(path, 0)
        if img is None:
            img = np.zeros(fallback_size)
            cv2.circle(img, (50, 50), 30, 1, -1)
        return img.astype(np.float64) / 255.0

    def add_noise(normalized_img, sigma):
        """
        Add Gaussian noise of sigma deviation to the normalized img.
        """
        noise = np.random.normal(0, sigma, normalized_img.shape)
        noisy_img = normalized_img + noise
        return np.clip(noisy_img, 0, 1)

    def mse(imgA, imgB):
        """
        Computes the Mean Squared Error between two images.
        """
        err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
        err /= float(imgA.shape[0] * imgA.shape[1])
        return err

    def psnr(imgA, imgB, max_pixel=1.0):
        """
        Computes the PSNR between two images.
        """
        mse_val = mse(imgA, imgB)
        if mse_val == 0:
            return float('inf')

        return 10 * np.log10((max_pixel ** 2) / mse_val)

    sigma /= 255.0

    # noise std
    original_image = load_image(img_path)
    noisy_image = add_noise(original_image.copy(), sigma)

    # denoising
    denoised_image = denoise_hashnlm(
        noisy_image.copy(), sigma, num_features, beta)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original (Clean)")
    plt.subplot(132)
    plt.imshow(noisy_image, cmap='gray')
    plt.title(
        f"Noisy MSE = {mse(original_image.copy()*255.0, noisy_image.copy()*255.0):.2f} | PSNR = {psnr(original_image, noisy_image):.4f}")
    plt.subplot(133)
    plt.imshow(denoised_image, cmap='gray')
    plt.title(
        f"Denoised MSE = {mse(original_image.copy()*255.0, denoised_image.copy()*255.0):.2f} | PSNR = {psnr(original_image, denoised_image):.4f}")

    plt.suptitle(
        f"Hashed NLM sigma={sigma:.3f}, beta={beta}, #features={num_features}")

    # plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def standard_comparison_hashed_nlm_zoomed(img_path, output_path, sigma=17, num_features=4, beta=0.88):
    """
        Zoomed standard comparison for hashed NLM.
        Please don't modify normalization of images, because they go crazy
    """
    from mcnlm.hashnlm import denoise_hashnlm
    import matplotlib.pyplot as plt
    import cv2

    # some functions required for this comparison
    # functions from utils mess up everything, numpy fails miserably
    def load_image(path, fallback_size=(100, 100)):
        """
        Load an image from the specified path.
        """
        img = cv2.imread(path, 0)
        if img is None:
            img = np.zeros(fallback_size)
            cv2.circle(img, (50, 50), 30, 1, -1)
        return img.astype(np.float64) / 255.0

    def add_noise(normalized_img, sigma):
        """
        Add Gaussian noise of sigma deviation to the normalized img.
        """
        noise = np.random.normal(0, sigma, normalized_img.shape)
        noisy_img = normalized_img + noise
        return np.clip(noisy_img, 0, 1)

    def mse(imgA, imgB):
        """
        Computes the Mean Squared Error between two images.
        """
        err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
        err /= float(imgA.shape[0] * imgA.shape[1])
        return err

    def psnr(imgA, imgB, max_pixel=1.0):
        """
        Computes the PSNR between two images.
        """
        mse_val = mse(imgA, imgB)
        if mse_val == 0:
            return float('inf')

        return 10 * np.log10((max_pixel ** 2) / mse_val)

    sigma /= 255.0

    # noise std
    original_image = load_image(img_path)
    noisy_image = add_noise(original_image.copy(), sigma)

    # denoising
    denoised_image = denoise_hashnlm(
        noisy_image.copy(), sigma, num_features, beta)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original (Clean)")
    plt.subplot(132)
    plt.imshow(noisy_image, cmap='gray')
    plt.title(
        f"Noisy MSE = {mse(original_image.copy()*255.0, noisy_image.copy()*255.0):.2f} | PSNR = {psnr(original_image, noisy_image):.4f}")
    plt.subplot(133)
    plt.imshow(denoised_image, cmap='gray')
    plt.title(
        f"Denoised MSE = {mse(original_image.copy()*255.0, denoised_image.copy()*255.0):.2f} | PSNR = {psnr(original_image, denoised_image):.4f}")

    plt.suptitle(
        f"Hashed NLM sigma={sigma:.3f}, beta={beta}, #features={num_features}")

    # plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    # if len(sys.argv) > 1:
    #     command = sys.argv[1]

    #     if command == 'compare_noise':
    #         print("Running noise estimation comparison...")
    #         noise_comparison_results()
    #     elif command == 'convergence':
    #         print("Running MC convergence analysis...")
    #         mc_convergence_results()
    #     elif command == 'mcnlm':
    #         print("Running MCNLM results...")
    #         results_mcnlm()
    #     elif command == 'nlm':
    #         print("Running NLM results...")
    #         results_naive_nlm()
    #     else:
    #         print(f"Unknown command: {command}")
    #         print("Available commands: compare_noise, convergence, mcnlm, nlm")
    # else:
    #     # Default behavior
    #     results_naive_nlm()
    #     mc_convergence_results()

    # Hashed NLM function to see how it works
    standard_comparison_hashed_nlm(
        "imgs/land.tiff", "../docs/res/hashednlm.pdf")

    # from mcnlm.utils import show_matches
    # show_matches('./imgs/clock.tiff', [(100, 100), (150, 200)], "../docs/res/robert_matches.pdf")
