import numpy as np
import scipy


def calculate_h(sigma=0.1, num_features=4, beta=0.88):
    """
    Calculate h (bin width) like in the paper.
    """
    return np.sqrt(2 * (sigma * sigma) * num_features * beta)


def extract_features_4(img):
    """
    Extract the 4 neighbours for every pixel. Padds the image to handle edges.
    """
    padded = np.pad(img, pad_width=1, mode='edge')

    f1 = padded[0:-2, 1:-1]  # Shift Down (up neighbour)
    f2 = padded[2:, 1:-1]  # Shift Up (down neighbour)
    f3 = padded[1:-1, 0:-2]  # Shift Right (left neighbour)
    f4 = padded[1:-1, 2:]   # Shift Left (right neighbour)

    return [f1, f2, f3, f4]


def extract_features_8(img):
    """
    Extracts the 8 surrounding neighbors for every pixel (3x3 neighborhood).
    """
    padded = np.pad(img, pad_width=1, mode='edge')

    # Shifts to get the 8 neighbors
    f1 = padded[0:-2, 1:-1]  # Top
    f2 = padded[2:,   1:-1]  # Bottom
    f3 = padded[1:-1, 0:-2]  # Left
    f4 = padded[1:-1, 2:]    # Right
    f5 = padded[0:-2, 0:-2]  # Top-Left
    f6 = padded[0:-2, 2:]    # Top-Right
    f7 = padded[2:,   0:-2]  # Bottom-Left
    f8 = padded[2:,   2:]    # Bottom-Right

    return [f1, f2, f3, f4, f5, f6, f7, f8]


def extract_features_6(img):
    """
    Extracts 4 neighbours and add spatiality by adding 2 more dimesnions, pixel positions.
    """
    rows, cols = img.shape
    padded = np.pad(img, pad_width=1, mode='edge')

    f1 = padded[0:-2, 1:-1]  # Up neighbour
    f2 = padded[2:,   1:-1]  # Down neighbour
    f3 = padded[1:-1, 0:-2]  # Left neighbour
    f4 = padded[1:-1, 2:]    # Right neighbour

    # Create a grid of X and Y coordinates
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))

    # Normalize coordinates to 0.0 - 1.0 range so they behave similarly intensity features during binning.
    f_x = x_coords.astype(float) / cols
    f_y = y_coords.astype(float) / rows

    return [f1, f2, f3, f4, f_x, f_y]


def create_hash_grid(img, feature_maps, bin_width):
    """
    Creates hash spaces.
    """

    indices = []
    grid_shapes = []

    for f in feature_maps:
        f_min = np.min(f)
        idx = np.round((f - f_min) / bin_width).astype(int)
        indices.append(idx)
        grid_shapes.append(np.max(idx) + 1)

    # H1: Counts how many pixels fall in each bin
    # Hf: Sums the intensities of pixels in each bin
    H1 = np.zeros(grid_shapes, dtype=float)
    Hf = np.zeros(grid_shapes, dtype=float)

    H1_indices = tuple(indices)
    np.add.at(H1, H1_indices, 1.0)
    np.add.at(Hf, H1_indices, img)

    return H1, Hf, indices


def blur_hash_space(H1, Hf):
    """
    Apply a Gaussian blur to the 4D grids.
    """
    # Sigma is 1 because our grid is already scaled

    # H1_prime: The denominator of the NLM equation
    H1_prime = scipy.ndimage.gaussian_filter(
        H1, sigma=1.0, mode='constant', cval=0.0)

    # Hf_prime: The numerator of the NLM equation
    Hf_prime = scipy.ndimage.gaussian_filter(
        Hf, sigma=1.0, mode='constant', cval=0.0)

    return H1_prime, Hf_prime


def reconstruct_image(H1_prime, Hf_prime, indices):
    """
    Reconstructs the denoised image using the blurred hash spaces.
    """
    numerator = Hf_prime[tuple(indices)]
    denominator = H1_prime[tuple(indices)]

    denoised_img = numerator / denominator

    return denoised_img


def denoise_hashnlm(noisy_image, sigma, num_features, beta):
    """
    Denoise an image with the hashnlm algorithm.
    """

    # Calculate bin width
    h = calculate_h(sigma, num_features, beta)

    # Extract features
    if num_features == 4:
        features = extract_features_4(noisy_image)
    elif num_features == 6:
        features = extract_features_6(noisy_image)
    elif num_features == 8:
        features = extract_features_8(noisy_image)

    # Create hash spaces
    H1, Hf, pixel_indices = create_hash_grid(noisy_image, features, h)

    # Blur hash spaces
    H1_prime, Hf_prime = blur_hash_space(H1, Hf)

    # Reconstruct
    denoised_image = reconstruct_image(H1_prime, Hf_prime, pixel_indices)

    # Optional blurring step
    # denoised_image = scipy.ndimage.gaussian_filter(
    #     denoised_image, sigma=0.5)
    return denoised_image
