import numpy as np


def bilinear_interpolation(image, orig_x, orig_y):
    """
      Perform bilinear interpolation for an entire grid of (orig_x, orig_y).

      Args:
          image (numpy.ndarray): Input image array of shape (H, W, C).
          orig_x (numpy.ndarray): X-coordinates for interpolation.
          orig_y (numpy.ndarray): Y-coordinates for interpolation.

      Returns:
          numpy.ndarray: Interpolated pixel values.
      """
    height, width, channels = image.shape

    # Integer and fractional parts of coordinates
    x1 = np.floor(orig_x).astype(int)
    y1 = np.floor(orig_y).astype(int)
    x2 = np.clip(x1 + 1, 0, width - 1)
    y2 = np.clip(y1 + 1, 0, height - 1)
    a = orig_x - x1
    b = orig_y - y1

    # Gather pixel values at four neighboring points
    q11 = image[y1, x1]  # Top-left
    q21 = image[y1, x2]  # Top-right
    q12 = image[y2, x1]  # Bottom-left
    q22 = image[y2, x2]  # Bottom-right

    # Bilinear interpolation (vectorized)
    top = (1 - a)[..., None] * q11 + a[..., None] * q21
    bottom = (1 - a)[..., None] * q12 + a[..., None] * q22
    interpolated_pixels = ((1 - b)[..., None] * top + b[..., None] * bottom).astype(np.uint8)

    return interpolated_pixels


def flipper(img, mode="horizontal"):
    """
    Flip an image horizontally, vertically, or both.

    Args:
        img (numpy.ndarray): Input image array.
        mode (str): Flip mode ('horizontal', 'vertical', or 'both').

    Returns:
        numpy.ndarray: Flipped image.
    """
    if mode not in {"horizontal", "vertical", "both"}:
        raise ValueError("Invalid mode. Choose 'horizontal', 'vertical', or 'both'.")
    flip_map = {"horizontal": (slice(None), slice(None, None, -1), slice(None)),
                "vertical": (slice(None, None, -1), slice(None), slice(None)),
                "both": (slice(None, None, -1), slice(None, None, -1), slice(None))}
    return img[flip_map[mode]]


def rotate_90(img, clockwise=True):
    """
        Rotate an image by 90 degrees.

        Args:
            img (numpy.ndarray): Input image array.
            clockwise (bool): Rotate clockwise if True, counterclockwise if False.

        Returns:
            numpy.ndarray: Rotated image.
        """
    if clockwise:
        return np.transpose(img, (1, 0, 2))[:, ::-1]
    else:
        return np.transpose(img, (1, 0, 2))


def rotate_180(img):
    """
    Rotate an image by 180 degrees.

    Args:
        img (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Rotated image.
    """
    return img[::-1, ::-1]


def rotate_degree(image, angle):
    """
        Rotate an image by a specified angle with optional interpolation.

        Args:
        image (numpy.ndarray): Input image array.
        angle (float): Rotation angle in degrees.
        interpolation (str): Interpolation method ('bilinear' or 'nearest').

        Returns:
        numpy.ndarray: Rotated image.
        """
    theta = np.radians(angle)

    # Handle grayscale and RGB images
    if image.ndim == 2:  # Grayscale
        height, width = image.shape
        colors = 1  # Single channel
    else:  # RGB
        height, width, colors = image.shape

    center_x, center_y = width / 2, height / 2  # More precise center calculation

    # Rotation matrix
    cos_b, sin_b = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_b, -sin_b],
                                [sin_b, cos_b]])

    # Compute new bounding box size
    corners = np.array([[-center_x, -center_y],
                        [center_x, -center_y],
                        [-center_x, center_y],
                        [center_x, center_y]])
    new_corners = np.dot(rotation_matrix, corners.T).T
    min_x, min_y = new_corners.min(axis=0)
    max_x, max_y = new_corners.max(axis=0)
    new_width, new_height = int(max_x - min_x), int(max_y - min_y)

    # Generate coordinate grid (Vectorized)
    y_indices, x_indices = np.meshgrid(np.arange(new_height), np.arange(new_width), indexing="ij")
    orig_cords = np.dot(rotation_matrix.T, np.stack([x_indices.ravel() + min_x, y_indices.ravel() + min_y]))

    orig_x = orig_cords[0, :] + center_x
    orig_y = orig_cords[1, :] + center_y

    # Filter valid coordinates
    valid_mask = (0 <= orig_x) & (orig_x < width) & (0 <= orig_y) & (orig_y < height)
    valid_x, valid_y = orig_x[valid_mask].astype(int), orig_y[valid_mask].astype(int)

    # Create new blank image
    if colors == 1:
        rotated_image = np.zeros((new_height, new_width), dtype=image.dtype)
        rotated_image[y_indices.ravel()[valid_mask], x_indices.ravel()[valid_mask]] = image[valid_y, valid_x]
    else:
        rotated_image = np.zeros((new_height, new_width, colors), dtype=image.dtype)
        rotated_image[y_indices.ravel()[valid_mask], x_indices.ravel()[valid_mask]] = image[valid_y, valid_x, :]

    return rotated_image


def rescale(image, new_width, new_height):
    """
        Resize an image using bilinear interpolation.

        Args:
            image (numpy.ndarray): Input image array.
            new_width (int): Target width.
            new_height (int): Target height.

        Returns:
            numpy.ndarray: Rescaled image.
        """
    height, width, channels = image.shape

    # Generate coordinate grid for the new image
    x_cords = np.linspace(0, width - 1, new_width)
    y_cords = np.linspace(0, height - 1, new_height)
    x_grid, y_grid = np.meshgrid(x_cords, y_cords)

    # Flatten grids for vectorized processing
    x_flat, y_flat = x_grid.ravel(), y_grid.ravel()

    # Apply bilinear interpolation for all points at once
    interpolated_pixels = bilinear_interpolation(image, x_flat, y_flat)

    # Reshape back to the new image shape
    return interpolated_pixels.reshape((new_height, new_width, channels))


def crop(img, x_start, y_start, height, width):
    """
       Crop a section of an image.

       Args:
           img (numpy.ndarray): Input image array.
           x_start (int): X-coordinate of the top-left corner.
           y_start (int): Y-coordinate of the top-left corner.
           height (int): Height of the cropped section.
           width (int): Width of the cropped section.

       Returns:
           numpy.ndarray: Cropped image section.
       """
    return img[y_start: y_start + height, x_start: x_start + width, :]
