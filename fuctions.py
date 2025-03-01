import numpy as np  # our main objective
import cv2  # opening and showing images from the system
import os


# change the path of the image to your image
image_path = r"hd-image.jpg"


def open_image(path):

    """checks if the image exists or not , if exists returns the image"""

    if os.path.exists(path):
        img = cv2.imread(image_path)
        return img


def show_image(image):

    """gets the image and show it"""
    # image = cv2.resize(img,(300,400))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1 -> Rotation of the image
def rotate_90(img, clockwise=True):
    # (H, W, 3) (Height × Width × Channels)

    if clockwise:
        return np.transpose(img, (1, 0, 2))[:, ::-1]
    else:
        return np.transpose(img, (1, 0, 2))


def rotate_180(img):
    return img[::-1, ::-1]


def bilinear_interpolation(image, orig_x, orig_y):
    """Performs bilinear interpolation for a given (orig_x, orig_y) coordinate."""
    height, width, channels = image.shape

    # Find the surrounding integer coordinates
    x1, y1 = int(np.floor(orig_x)), int(np.floor(orig_y))
    x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

    # Check if within bounds
    if 0 <= x1 < width - 1 and 0 <= y1 < height - 1:
        a = orig_x - x1  # Fractional part of x
        b = orig_y - y1  # Fractional part of y

        # Fetch the four neighboring pixels
        i11 = image[y1, x1]
        i12 = image[y2, x1]
        i21 = image[y1, x2]
        i22 = image[y2, x2]

        # Compute bilinear interpolation
        interpolated_pixel = (
                i11 * (1 - a) * (1 - b) +
                i21 * a * (1 - b) +
                i12 * (1 - a) * b +
                i22 * a * b
        )
        return interpolated_pixel.astype(np.uint8)
    else:
        return np.zeros(3, dtype=np.uint8)  # Return black if out of bounds

def rotate_degree(image, angle):
    # change the angle to radian
    theta = np.radians(angle)

    # find the image center
    height, width, colors = image.shape
    center_height, center_width = height // 2, width // 2

    # the rotation matrix
    # |cos(b)  -sin(b)|
    # |sin(b)   cos(b)|
    cos_b, sin_b = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_b, -sin_b],
                                [sin_b, cos_b]])

    # corners
    # generally the image's top-left corner is at 0,0 .so, we move the image to get it to the center
    # so that the center is at 0,0  so the new coordinates of the corners will be
    corners = np.array([[-center_width, - center_height],  # top-left corner
                        [center_width, -center_height],  # top-right corner
                        [-center_width, center_height],  # bottom-left corner
                        [center_width, center_height]   # bottom-right corner
                        ])
    # new corners
    new_corners = np.dot(rotation_matrix, corners.T).T

    # now the image is rotated (dimond rotated square)but the output still should be a square, so we create new box to
    # enclose the rotated image
    # finding maximum and minimum corners
    min_x, min_y = new_corners.min(axis=0)
    max_x, max_y = new_corners.max(axis=0)
    # calculating the new width and height
    new_width = int(max_x - min_x)
    new_height = int(max_y - min_y)

    # create a new image
    rotated_image = np.zeros((new_height, new_width, colors), dtype=np.uint8)  # all zero i.e. the whole image is black

    # change the color of the image
    # mapping the rotated pixels to the new image
    for y in range(new_height):
        for x in range(new_width):
            # find the original coordinates of the image
            # rotating the matrix to map
            orig_x, orig_y = np.dot(rotation_matrix.T, [x + min_x, y + min_y])
            orig_x, orig_y = int(orig_x + center_width), int(orig_y + center_height)

            rotated_image[y,x] =bilinear_interpolation(image,orig_x, orig_y)

    return rotated_image


# flipping the image
def flipper(img, mode="horizontal"):
    """
    mode: "horizontal", "vertical", or "both"
    """
    if mode == "horizontal":
        # returning row as it is, column flipped , and colors the same
        return img[:, ::-1, :]
    elif mode == "vertical":
        # returning column as it is, row flipped , and colors the same
        return img[::-1, :, :]
    elif mode == "both":
        # returning row and column flipped , and colors the same
        return img[::-1, ::-1, :]
    else:
        raise ValueError("Invalid mode. Choose 'horizontal', 'vertical', or 'both'.")


# going with simple crop functionality
def crop(img, x_start, y_start , height, width):
    return img[y_start: y_start + height, x_start : x_start+ width, :]


def rescale(image , new_width , new_height):
    height, width, color = image.shape
    scaled_image = np.zeros((new_height, new_width, color), dtype=np.uint8)
    x_scale = width / new_width
    y_scale = height / new_height

    for y in range(new_height):
        for x in range(new_width):
            # Map new (x, y) to original coordinates
            orig_x = x * x_scale
            orig_y = y * y_scale
            scaled_image[y, x] = bilinear_interpolation(image, orig_x, orig_y)
    return scaled_image

# grayscale using luminosity
def grayscale_luminosity(img):
    weights = np.array([0.2989, 0.5870, 0.1140])
    return  np.dot(img[...,:3], weights).astype(np.uint8)

# creating a binary white and black image
def binary_iamge(img , threshold= 128):
    # convert the image to grayScale first
    if img.ndim ==3 and img.shape[-1] >= 3:
        gray = grayscale_luminosity(img)
    else:
        gray = img.astype(np.float32)
    return np.where( gray >= threshold, 255, 0).astype(np.uint8)

# negation
def Negation(img):
    """simply retun the complement of the image. 255 is the maximum number, so minus the image from it """
    return 255 - img



# adjustion the brightness of the image
def adjust_brighness(img , brightness = 10):
    """simply add the brightness into the image if positive then the image becomes bright
    else it becomes dark and also check if the value is between 0 and 255"""

    img = img.astype(np.int16) + brightness
    img = np.clip(img, 0,255)
    return img.astype(np.uint8)

# contrast
def adjust_contrast(img, factor):
    """Contrast enhancement adjusts the difference between dark and bright areas.
    It is achieved by scaling pixel values around a midpoint (usually 128).
    Formula:
    New Pixel=(Pixel−128)×Contrast Factor+128

    Contrast Factor > 1 → Increases contrast (makes darks darker and brights brighter).
    Contrast Factor < 1 → Decreases contrast (makes the image more uniform)."""

    img = (img.astype(np.float32) -128) * factor +128
    return np.clip(img,0,255).astype(np.uint8)


# kernel for the blur
def gaussian_kernel(size = 3,sigma =1.0):
    """Generate a Gaussian kernel."""
    ax =  np.linspace(-(size //2), size // 2,size)
    xx, yy = np.meshgrid(ax,ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize

# convolution
def convolve(img,kernel):
    """Apply a convolution with a given kernel."""
    h, w = img.shape[:2]
    k_size = kernel.shape[0]
    pad = k_size //2
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant') # Zero padding
    result = np.zeros((h, w,img.shape[2]), dtype=np.float32)

    for c in range(img.shape[2]):
        for i in range(h):
            for j in range(w):
                result[i, j, c] = np.sum(img_padded[i:i+k_size, j:j+k_size,c] * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_blur(img,size=3,sigma=1.0):
    """
    A Gaussian kernel is a weighted matrix where nearby pixels contribute more than distant ones.
    Example of a 3×3 Gaussian kernel (with σ=1.0):
                      | 1    2    1 |
                1/16* |  2   4    2 |
                      | 1    2    1|
    """
    kernel = gaussian_kernel(size, sigma)
    return convolve(img, kernel)


def compute_gradient_magnitude_direction(gx, gy):
    """Compute gradient magnitude and direction using Sobel gradients."""
    magnitude = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
    direction = np.arctan2(gy, gx)  # Compute gradient angles in radians

    return np.clip(magnitude, 0, 255).astype(np.uint8), direction

def edge_detection(img):
    """Apply Sobel edge detection after converting to grayscale."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    img = gaussian_blur(img, 5, 1)
    img_gray = grayscale_luminosity(img)  # Use your function
    gx = convolve(img_gray[..., np.newaxis], sobel_x).squeeze()  # Apply Sobel X
    gy = convolve(img_gray[..., np.newaxis], sobel_y).squeeze() # apply sobel Y

    edge_magnitude,direction = compute_gradient_magnitude_direction(gx,gy)

    return edge_magnitude,direction


# applying non-maximum suppression
def non_maximum_suppression(gradient_tuple):
    """Suppress non-maximum pixels in the gradient direction."""
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

# Hysteresis Thresholding
def hysteresis_thresholding(img, low_thresh, high_thresh):
    """Apply hysteresis thresholding to keep strong edges and discard weak ones."""
    strong = 255
    weak = 50  # Arbitrary weak value
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Strong edges
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


def canny_edge_detection(img, low_thresh=50, high_thresh=100):
    """Complete Canny Edge Detection pipeline."""
    gradient = edge_detection(img)
    nms_edges = non_maximum_suppression(gradient)
    final_edges = hysteresis_thresholding(nms_edges, low_thresh, high_thresh)

    return final_edges

img = open_image(image_path)
# img =rescale(img, 500,800)
# img= canny_edge_detection(img)
img = cv2.resize(img,(500,300))
show_image(img)

