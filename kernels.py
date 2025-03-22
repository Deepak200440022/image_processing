import numpy as np


def gaussian_kernel(size=3, sigma=1.0):
    """
    Generate a normalized 2D Gaussian kernel.

    Parameters:
    size (int): Size of the kernel (odd number preferred).
    sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
    np.ndarray: Normalized 2D Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)  # Define axis range
    xx, yy = np.meshgrid(ax, ax, indexing='xy')   # Create coordinate grid
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)  # Compute Gaussian function
    return kernel / np.sum(kernel)  # Normalize to sum=1


def convolve(img, ker, mode="constant"):
    """
       Apply a convolution operation on an image using a given kernel.

       Parameters:
       img (np.ndarray): Input image (grayscale or color).
       ker (np.ndarray): Convolution kernel.
       mode (str): Padding mode ('constant', 'reflect', 'edge', etc.).

       Returns:
       np.ndarray: Convolved image with the same shape as the input.
       """
    ker = np.rot90(ker, 2)  # Rotate kernel by 180° for proper convolution
    h, w = img.shape[:2]  # Get image dimensions
    k_h, k_w = ker.shape[:2]  # Get kernel dimensions
    pad_h = k_h // 2  # Compute padding height
    pad_w = k_w // 2  # Compute padding width
    result = None
    if img.ndim == 2:  # Grayscale image
        img_padded = np.pad(img, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode=mode)  # Apply padding to the image

        # Create a sliding window view of the padded image
        strided_shape = (h, w, k_h, k_w)
        strided_strides = img_padded.strides[:2] + img_padded.strides[:2]
        img_windows = np.lib.stride_tricks.as_strided(img_padded, shape=strided_shape, strides=strided_strides)
        # Perform convolution using Einstein summation (memory efficient)
        result = np.einsum('ijkl,kl->ij', img_windows, ker)

    elif img.ndim == 3:  # Color image (RGB)
        # Apply padding to each channel separately
        img_padded = np.pad(img, pad_width=((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=mode)
        # Create a sliding window view for all channels
        strided_shape = (h, w, k_h, k_w, img.shape[2])
        strided_strides = img_padded.strides[:2] + img_padded.strides[:2] + img_padded.strides[2:]
        img_windows = np.lib.stride_tricks.as_strided(img_padded, shape=strided_shape, strides=strided_strides)
        # Perform convolution using Einstein summation
        result = np.einsum('ijklc,kl->ijc', img_windows, ker) if (ker.ndim == 2)\
            else np.einsum('ijklc,klc->ijc', img_windows, ker)

    return np.clip(result, 0, 255).astype(np.uint8)  # Clip values to valid range and convert to uint8
