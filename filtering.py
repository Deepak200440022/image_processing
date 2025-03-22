import numpy as np
import thresholding
from kernels import convolve
from image_IO import gaussian_blur, grayscale_luminosity


def edge_detection(img):
    """
       Apply Sobel edge detection after converting the image to grayscale.

       Steps:
       1. Apply Gaussian blur to smooth the image and reduce noise.
       2. Convert the image to grayscale using luminosity method.
       3. Apply the Sobel operator in both X and Y directions.
       4. Compute the gradient magnitude and direction.

       Args:
           img (numpy.ndarray): Input image.

       Returns:
           tuple: Edge magnitude and gradient direction.
       """
    # Define Sobel kernels for horizontal and vertical edge detection
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    # Apply Gaussian blur to reduce noise before edge detection
    img = gaussian_blur(img, 3, 1)  # Smooth before detecting edges
    # Convert the image to grayscale
    img_gray = grayscale_luminosity(img)
    # Compute gradients using Sobel filters
    gx = convolve(img_gray, sobel_x)  # Sobel X
    gy = convolve(img_gray, sobel_y)  # Sobel Y
    # Compute gradient magnitude and direction
    edge_magnitude, direction = thresholding.compute_gradient_magnitude_direction(gx, gy)

    return edge_magnitude, direction


def canny_edge_detection(img, low_thresh=50, high_thresh=100):
    """
        Perform Canny edge detection using the following steps:

        1. Compute the gradient magnitude and direction using Sobel operators.
        2. Apply Non-Maximum Suppression (NMS) to thin edges.
        3. Use double thresholding to classify strong, weak, and non-edges.
        4. Perform edge tracking by hysteresis to finalize edge selection.

        Args:
            img (numpy.ndarray): Input image.
            low_thresh (int): Lower threshold for hysteresis.
            high_thresh (int): Upper threshold for hysteresis.

        Returns:
            numpy.ndarray: Binary image with detected edges.
        """
    # Compute gradient magnitude and direction
    gradient = edge_detection(img)
    # Apply Non-Maximum Suppression to thin edges
    nms_edges = thresholding.non_maximum_suppression(gradient)
    # Apply double thresholding and hysteresis to finalize edges
    final_edges = thresholding.hysteresis_thresholding(nms_edges, low_thresh, high_thresh)

    return final_edges


def sharpen(img):
    """
        Sharpen the image using the Laplacian filter.

        The Laplacian kernel enhances edges by emphasizing regions with rapid intensity changes.

        Kernel used:
            |  0  -1   0  |
            | -1   5  -1  |
            |  0  -1   0  |

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Sharpened image.
        """
    # Define Laplacian sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Apply convolution to enhance edges
    return convolve(img, kernel)
