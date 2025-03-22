import numpy as np


def compress_image(img, levels=16):
    """
    Reduce the number of intensity levels for lossy compression.

    Parameters:
    img (numpy.ndarray): Input image in uint8 format.
    levels (int): Number of intensity levels to reduce to (default is 16).

    Returns:
    numpy.ndarray: Compressed image with reduced intensity levels.
    """
    factor = 256 // levels  # Compute the scaling factor for intensity reduction
    compressed_img = (img // factor) * factor  # Quantize pixel values to nearest level
    return compressed_img.astype(np.uint8)


def erase(img, x_start, y_start, height, width):
    """
       Erase a rectangular region of an image by setting pixel values to zero (black).

       Parameters:
       img (numpy.ndarray): Input image in uint8 format.
       x_start (int): X-coordinate of the top-left corner of the rectangle.
       y_start (int): Y-coordinate of the top-left corner of the rectangle.
       height (int): Height of the rectangle.
       width (int): Width of the rectangle.

       Returns:
       numpy.ndarray: Modified image with the specified region erased.
       """
    img[y_start: y_start + height, x_start: x_start + width, :] = 0
    return img


def color_filter(img, color="blue"):
    """
        Apply a color filter by removing one of the RGB channels.

        Parameters:
        img (numpy.ndarray): Input image in BGR format (as used by OpenCV).
        color (str): Color channel to remove ("blue", "green", or "red"). Default is "blue".

        Returns:
        numpy.ndarray: Modified image with the specified color channel removed.

        Raises:
        ValueError: If an invalid color is provided.
        """
    color_dict = {"blue": 0, "green": 1, "red": 2}  # Mapping of colors to channel indices
    if color in color_dict:
        img[:, :, color_dict[color]] = 0  # Set the selected color channel to zero
    else:
        raise ValueError("Color not found. Choose from 'blue', 'green', or 'red'.")
    return img
