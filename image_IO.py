import cv2
import numpy as np
import os
from kernels import gaussian_kernel, convolve


def open_image(path):
    """
     Checks if the image file exists at the given path.
     If it exists, reads and returns the image using OpenCV.

     Parameters:
         path (str): The file path of the image.

     Returns:
         numpy.ndarray: The loaded image.
     """
    if os.path.exists(path):
        img = cv2.imread(path)
        return img


def show_image(image):
    """
       Displays an image using OpenCV.

       Parameters:
           image (numpy.ndarray): The image to be displayed.
       """
    # Display the image in a window
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_blur(img, size=3, sigma=1.0, grayscale=True):
    """
       Applies Gaussian blur to an image using a generated Gaussian kernel.

       Parameters:
           img (numpy.ndarray): The input image.
           size (int, optional): The kernel size (must be an odd number). Default is 3.
           sigma (float, optional): The standard deviation for the Gaussian function. Default is 1.0.
           grayscale (bool, optional): If True, converts the image to grayscale before applying the blur.
            Default is True.

       Returns:
           numpy.ndarray: The blurred image.
       """
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(size, sigma)

    # Convert to grayscale if required
    if grayscale:
        img = grayscale_luminosity(img)
    # Apply convolution with Gaussian kernel
    blurred = convolve(img, kernel, mode="reflect")

    return blurred


def save_image(image, path):
    """
       Saves an image to the specified path using OpenCV.

       Parameters:
           image (numpy.ndarray): The image to save.
           path (str): The file path where the image will be saved.
       """
    cv2.imwrite(path, image)


def grayscale_luminosity(img):
    """
    Converts an RGB image to grayscale using the luminosity method.

    The luminosity method accounts for human perception, giving more weight to green.

    Parameters:
        img (numpy.ndarray): The input RGB image.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    if img.ndim == 3:   # Check if the image has 3 channels (RGB)
        img = np.dot(img[..., :3], [0.2989, 0.587, 0.114])  # Apply luminosity formula
    return img.astype(np.uint8)  # Ensure output is 2D and of correct type


def adjust_contrast(img, factor):
    """
       Adjusts the contrast of an image by scaling pixel values around a midpoint (128).

       Formula:
           New Pixel = (Pixel − 128) × Contrast Factor + 128

       Parameters:
           img (numpy.ndarray): The input image.
           factor (float): The contrast adjustment factor.
                           >1 increases contrast, <1 decreases contrast.

       Returns:
           numpy.ndarray: The contrast-adjusted image.
       """
    img = (img.astype(np.float32) - 128) * factor + 128
    return np.clip(img, 0, 255).astype(np.uint8)  # Ensure pixel values remain valid


def adjust_brightness(img, brightness=10):
    """
       Adjusts the brightness of an image by adding a constant value to all pixel intensities.

       Parameters:
           img (numpy.ndarray): The input image.
           brightness (int, optional): The value to be added to pixel intensities.
                                       Positive values increase brightness, negative values decrease it. Default is 10.

       Returns:
           numpy.ndarray: The brightness-adjusted image.
       """

    img = img.astype(np.int16) + brightness  # Use int16 to prevent overflow
    img = np.clip(img, 0, 255)  # Ensure pixel values remain within valid range
    return img.astype(np.uint8)
