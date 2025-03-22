import numpy as np
from image_IO import grayscale_luminosity


def hysteresis_thresholding(img, low_thresh, high_thresh):
    """
    Apply hysteresis thresholding to keep strong edges and discard weak ones.

    Parameters:
        img (numpy.ndarray): Grayscale image.
        low_thresh (int): Lower threshold for weak edges.
        high_thresh (int): Upper threshold for strong edges.

    Returns:
        numpy.ndarray: Image after hysteresis thresholding.
    """
    strong = 255
    weak = 50  # Arbitrary weak value
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Identify strong and weak edges
    strong_edges = img >= high_thresh
    weak_edges = (img >= low_thresh) & (img < high_thresh)

    result[strong_edges] = strong
    result[weak_edges] = weak

    # Iterate over weak pixels and check if connected to strong edges
    for i in range(1, h-1):
        for j in range(1, w-1):
            if result[i, j] == weak:
                # Check 8 neighbors for a strong edge
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0  # Suppress weak edge

    return result


def binary_image(img, threshold=128):
    """
       Convert an image to a binary image using a given threshold.

       Parameters:
           img (numpy.ndarray): Input image (grayscale or RGB).
           threshold (int): Threshold value for binarization.

       Returns:
           numpy.ndarray: Binary image with pixel values 0 or 255.
       """
    # Convert the image to grayscale if it is in RGB format
    if img.ndim == 3 and img.shape[-1] >= 3:
        gray = grayscale_luminosity(img)
    else:
        gray = img.astype(np.float32)
    return np.where(gray >= threshold, 255, 0).astype(np.uint8)


def compute_gradient_magnitude_direction(gx, gy):
    """
       Compute the gradient magnitude and direction from Sobel gradients.

       Parameters:
           gx (numpy.ndarray): Gradient in the x-direction.
           gy (numpy.ndarray): Gradient in the y-direction.

       Returns:
           tuple: (Gradient magnitude, Gradient direction in radians)
       """
    magnitude = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
    direction = np.arctan2(gy, gx)   # Compute gradient angles in radians

    return np.clip(magnitude, 0, 255).astype(np.uint8), direction


def non_maximum_suppression(gradient_tuple):
    """
     Suppress non-maximum pixels in the gradient direction to thin edges.

     Parameters:
         gradient_tuple (tuple): (Gradient magnitude, Gradient direction in radians).

     Returns:
         numpy.ndarray: Image with suppressed non-maximum edges.
     """
    magnitude, direction = gradient_tuple  # Unpack tuple
    h, w = magnitude.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Convert angles to nearest 0°, 45°, 90°, or 135°
    angle = np.rad2deg(direction) % 180  # Convert radians to degrees & normalize

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q, r = 255, 255  # Default high values (to be replaced)

            # Determine neighboring pixels to compare
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]  # Right
                r = magnitude[i, j - 1]  # Left
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]  # Bottom-left
                r = magnitude[i - 1, j + 1]  # Top-right
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]  # Bottom
                r = magnitude[i - 1, j]  # Top
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]  # Top-left
                r = magnitude[i + 1, j + 1]  # Bottom-right

            # Suppress non-maximum values
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                result[i, j] = magnitude[i, j]

    return result


def negation(img):
    """
     Compute the complement (negative) of an image.

     Parameters:
         img (numpy.ndarray): Input image.

     Returns:
         numpy.ndarray: Negative of the input image.
     """
    return 255 - img
