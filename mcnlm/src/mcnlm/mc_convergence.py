import numpy as np
import matplotlib.pyplot as plt

from mcnlm.utils import mse, add_gaussian_noise, load_image, psnr, estimate_noise
import mcnlm.mc_nlm as mc_nlm
import mcnlm.naive_nlm as naive_nlm


def run_mc_convergence(image_path, xis, known_sigma=True, seed: int | None = None, deterministic: bool = False):
    """
    Run MCNLM for different sampling probabilities and compute MSE between MC method and naive NLM.
    """
    image = load_image(image_path)
    sigma = 17
    if seed is not None:
        np.random.seed(seed)
    noisy = add_gaussian_noise(
        image * 255, sigma=sigma).astype(np.float32) / 255.0

    known_sigma = sigma if known_sigma else estimate_noise(noisy * 255)
    print(f"Using sigma = {known_sigma} for denoising")
    nlm_params = naive_nlm.NLMParams(
        sigma=sigma / 255.0,
        h_factor=0.4,
        patch_radius=2,
        search_radius=10
    )

    # compute naive NLM
    print("Computing naive NLM...")
    nlm_ref = naive_nlm.test_naive_nlm(noisy, nlm_params) * 255.0
    naive_clean_mse = mse(nlm_ref, image)
    native_clean_psnr = psnr(nlm_ref, image)

    print(
        f"Naive NLM MSE vs clean: {naive_clean_mse}, PSNR vs clean: {native_clean_psnr}")

    mc_clean_mse = []
    mc_clean_psnr = []
    print("Running Monte-Carlo with different sampling probabilities...")

    SEARCH_RADIUS = 10
    print(
        f"MC-NLM with search window {2 * SEARCH_RADIUS + 1}×{2 * SEARCH_RADIUS + 1}")

    for idx, xi in enumerate(xis):
        print(f"sampling_prob = {xi}")

        mc_params = mc_nlm.MCNLMParams(
            sigma=sigma / 255.0,
            h_factor=0.4,
            patch_size=5,
            search_radius=SEARCH_RADIUS,
            spatial_sigma=1e10,
            sampling_prob=xi
        )
        mc_seed = None
        if seed is not None:
            mc_seed = seed + 1000 + idx
        mc = mc_nlm.test_mcnlm(noisy, mc_params, deterministic=deterministic, seed=mc_seed) * 255.0
        mc_clean_mse.append(mse(mc, image))
        mc_clean_psnr.append(psnr(mc, image))
        print(
            f"  MSE vs clean: {mc_clean_mse[-1]}, PSNR vs clean: {mc_clean_psnr[-1]}")

    return xis, mc_clean_mse, mc_clean_psnr, naive_clean_mse, native_clean_psnr


def mc_convergence(
    image_path,
    output_path1,
    output_path2,
    seed: int | None = None,
    deterministic: bool = False,
    show: bool = True,
):
    probs = np.linspace(0, 1, 13)
    # probs = [1]
    print('Testing convergence for probs: ', probs)
    xis, mc_clean_errors, mc_clean_psnr, naive_mse, naive_psnr = run_mc_convergence(
        image_path, xis=probs, seed=seed, deterministic=deterministic)

    plt.figure(figsize=(7, 5))

    # MC-NLM MSE
    plt.plot(xis, mc_clean_errors, 'o-', label='MC-NLM vs clean')
    plt.axhline(y=naive_mse, color='r', linestyle='--',
                label=f'Naive NLM MSE = {naive_mse}')

    plt.xlabel('Sampling probs')
    plt.ylabel('MSE vs clean image')
    plt.title('MC-NLM Denoising vs Naive NLM')
    plt.xticks(xis)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path1, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    # MC-NLM PSNR
    plt.figure(figsize=(7, 5))
    plt.plot(xis, mc_clean_psnr, 'o-', label='MC-NLM vs clean')
    plt.axhline(y=naive_psnr, color='r', linestyle='--',
                label=f'Naive NLM PSNR = {naive_psnr}')

    plt.xlabel('Sampling probs')
    plt.ylabel('PSNR vs clean image')
    plt.title('MC-NLM Denoising vs Naive NLM')
    plt.xticks(xis)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path2, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def compare_noise_estimation(
    image_path,
    output_path_visual,
    output_path_mse,
    output_path_psnr,
    seed: int | None = None,
    deterministic: bool = False,
    show: bool = False,
):
    """
    Compare MCNLM performance with known vs estimated noise.
    Creates side-by-side comparison and convergence plots.
    """
    image = load_image(image_path)
    true_sigma = 17
    if seed is not None:
        np.random.seed(seed)
    noisy = add_gaussian_noise(
        image * 255, sigma=true_sigma).astype(np.float32) / 255.0

    estimated_sigma = estimate_noise(noisy * 255)

    print(f"True sigma: {true_sigma}, Estimated sigma: {estimated_sigma:.2f}")

    # Test with p=1 for visual comparison
    print("\nTesting with p=1 for visual comparison...")
    params_known = mc_nlm.MCNLMParams(
        sigma=true_sigma / 255.0,
        h_factor=0.8,
        patch_size=5,
        search_radius=10,
        spatial_sigma=1e10,
        sampling_prob=1.0
    )

    params_estimated = mc_nlm.MCNLMParams(
        sigma=estimated_sigma / 255.0,
        h_factor=0.8,
        patch_size=5,
        search_radius=10,
        spatial_sigma=1e10,
        sampling_prob=1.0
    )

    mc_seed = None
    if seed is not None:
        mc_seed = seed + 2000
    denoised_known = mc_nlm.test_mcnlm(noisy, params_known, deterministic=deterministic, seed=mc_seed) * 255.0
    denoised_estimated = mc_nlm.test_mcnlm(noisy, params_estimated, deterministic=deterministic, seed=mc_seed) * 255.0

    # Visual comparison plot
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    noisy *= 255.0

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Clean Image')
    axs[0].axis('off')

    axs[1].imshow(noisy, cmap='gray')
    axs[1].set_title(f'Noisy Image\nMSE = {mse(image.copy(), noisy):.4f}')
    axs[1].axis('off')

    axs[2].imshow(denoised_known, cmap='gray')
    axs[2].set_title(
        f'Known σ = {true_sigma}\nMSE = {mse(image.copy(), denoised_known):.4f}')
    axs[2].axis('off')

    axs[3].imshow(denoised_estimated, cmap='gray')
    axs[3].set_title(
        f'Estimated σ = {estimated_sigma:.2f}\nMSE = {mse(image.copy(), denoised_estimated):.4f}')
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path_visual, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Visual comparison saved to {output_path_visual}")
