import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mcnlm.utils import add_gaussian_noise, load_image, psnr, mse
import mcnlm.mc_nlm as mc_nlm
from mcnlm.kdtree import run_kdtree_naive


def compare_all_methods(
    image_path: str,
    output_path: str,
    zoom: tuple | None = None,
    sampling_prob: float = 0.5,
    seed: int | None = None,
    deterministic: bool = False,
    show: bool = False,
):
    # Load and prepare image
    image = load_image(image_path)
    sigma = 17.0
    if seed is not None:
        np.random.seed(seed)
    noisy = add_gaussian_noise(image * 255, sigma) / 255.0
    image /= 255.0

    # Compute MCNLM denoising
    print("Computing MCNLM denoising...")
    mc_params = mc_nlm.MCNLMParams(
        sigma=sigma / 255.0,
        h_factor=0.4,
        patch_size=5,
        search_radius=10,
        spatial_sigma=1e10,
        sampling_prob=sampling_prob
    )
    mc_seed = None
    if seed is not None:
        mc_seed = seed + 100
    mcnlm_denoised = mc_nlm.test_mcnlm(noisy, mc_params, deterministic=deterministic, seed=mc_seed)
    
    # Compute KD-Tree denoising
    print("Computing KD-Tree denoising...")
    patch_size = 5
    padded_noisy = np.pad(noisy, patch_size // 2, mode='reflect')
    kdtree_denoised = run_kdtree_naive(padded_noisy, patch_size, sigma=sigma/255.0)
    pad = patch_size // 2
    kdtree_denoised = kdtree_denoised[pad:-pad, pad:-pad]
    
    # Convert to 0-255 range for display
    noisy_display = noisy * 255.0
    mcnlm_display = mcnlm_denoised * 255.0
    kdtree_display = kdtree_denoised * 255.0

    image *= 255.0

    mse_noisy = mse(image.copy(), noisy_display)
    mse_mcnlm = mse(image.copy(), mcnlm_display)
    mse_kdtree = mse(image.copy(), kdtree_display)
    
    # Compute PSNR values
    psnr_noisy = psnr(image.copy(), noisy_display)
    psnr_mcnlm = psnr(image.copy(), mcnlm_display)
    psnr_kdtree = psnr(image.copy(), kdtree_display)
    
    # Set up the figure
    if zoom is not None:
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        x0, y0, w, h = zoom
        x1, y1 = x0 + w, y0 + h
    else:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs = axs.reshape(1, -1)
    
    # First row: full images
    images = [
        (noisy_display, f'Noisy\nMSE={mse_noisy:.2f}\nPSNR={psnr_noisy:.2f} dB'),
        (mcnlm_display, f'MCNLM p={sampling_prob}\nMSE={mse_mcnlm:.2f}\nPSNR={psnr_mcnlm:.2f} dB'),
        (kdtree_display, f'K-D Tree Naive\nMSE={mse_kdtree:.2f}\nPSNR={psnr_kdtree:.2f} dB'),
    ]
    
    # Add clean image as first column
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Clean')
    axs[0, 0].axis('off')
    
    for col, (img, title) in enumerate(images, start=1):
        axs[0, col].imshow(img, cmap='gray')
        axs[0, col].set_title(title)
        axs[0, col].axis('off')
        
        # Draw zoom box if specified
        if zoom is not None:
            from matplotlib.patches import Rectangle
            rect = Rectangle((x0, y0), w, h,
                            edgecolor='red',
                            facecolor='none',
                            linewidth=2)
            axs[0, col].add_patch(rect)
    
    # Add zoom box to clean image as well
    if zoom is not None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((x0, y0), w, h,
                        edgecolor='red',
                        facecolor='none',
                        linewidth=2)
        axs[0, 0].add_patch(rect)
    
    # Second row: zoomed regions (if zoom is specified)
    if zoom is not None:
        # Clean zoom
        axs[1, 0].imshow(image[y0:y1, x0:x1], cmap='gray')
        axs[1, 0].set_title('Clean (zoom)')
        axs[1, 0].axis('off')
        
        # Noisy zoom
        axs[1, 1].imshow(noisy_display[y0:y1, x0:x1], cmap='gray')
        axs[1, 1].set_title('Noisy (zoom)')
        axs[1, 1].axis('off')
        
        # MCNLM zoom
        axs[1, 2].imshow(mcnlm_display[y0:y1, x0:x1], cmap='gray')
        axs[1, 2].set_title('MCNLM (zoom)')
        axs[1, 2].axis('off')
        
        # KD-Tree zoom
        axs[1, 3].imshow(kdtree_display[y0:y1, x0:x1], cmap='gray')
        axs[1, 3].set_title('K-D Tree Naive (zoom)')
        axs[1, 3].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"Comparison saved to {output_path}")
    print(f"PSNR - Noisy: {psnr_noisy:.2f} dB, MCNLM: {psnr_mcnlm:.2f} dB, KD-Tree: {psnr_kdtree:.2f} dB")
