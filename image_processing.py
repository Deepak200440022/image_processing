import cv2
import image_IO
import thresholding
import tkinter as tk
from tkinter import filedialog
import filtering
import utils
import geometric


def apply_processing(original, key):
    """
        Applies the corresponding image transformation based on the pressed key.

        Parameters:
        original (numpy.ndarray): The original image to be processed.
        key (int): The key pressed by the user.

        Returns:
        numpy.ndarray: The processed image after applying the transformation.
        """
    # Dictionary mapping keys to corresponding processing functions
    key_mapping = {
        ord('1'): lambda img: image_IO.grayscale_luminosity(img),
        ord('2'): lambda img: thresholding.binary_image(img, 128),
        ord('3'): lambda img: thresholding.negation(img),
        ord('5'): lambda img: filtering.sharpen(img),
        ord('6'): lambda img: utils.compress_image(img, levels=16),
        ord('7'): lambda img: utils.erase(img, 50, 50, 100, 100),
        ord('8'): lambda img: utils.color_filter(img, color="blue"),
        ord('r'): lambda img: geometric.rotate_90(img),
        ord('t'): lambda img: geometric.rotate_180(img),
        ord('y'): lambda img: geometric.rotate_degree(img, 45),
        ord('f'): lambda img: geometric.flipper(img),
        ord('c'): lambda img: geometric.crop(img, 50, 50, 200, 200),
        ord('s'): lambda img: geometric.rescale(img, 350, 300),
        ord('b'): lambda img: image_IO.adjust_brightness(img, 50),
        ord('n'): lambda img: image_IO.adjust_contrast(img, 1.5),
        ord('g'): lambda img: image_IO.gaussian_blur(img, 5, 1.5, grayscale=False),
        ord('e'): lambda img: filtering.edge_detection(img)[0],  # Extract processed image
        ord('h'): lambda img: filtering.canny_edge_detection(img, 50, 150),

    }
    # Apply the transformation if key exists in mapping, else return original image
    return key_mapping.get(key, lambda img: img)(original)  # Default to original if key not found


def process_image_interactive(image_path):
    """
      Opens an image and allows interactive processing through key presses.

      Parameters:
      image_path (str): Path to the image file.
      """

    original = image_IO.open_image(image_path)
    base_original = original.copy()  # Keep a copy of the original image
    processed = base_original.copy()
    # Display original and processed image windows
    cv2.imshow("Original Image", original)
    cv2.imshow("Processed Image", processed)

    while True:
        key = cv2.waitKey(0) & 0xFF  # Wait for key press

        if key == ord('0'):  # Reset to original
            processed = base_original.copy()
            original = base_original.copy()
        elif key == 27:  # ESC Key to Exit
            break
        elif key in [ord(k) for k in '1235678rtyfcsbnghe']:
            processed = apply_processing(original, key)  # Apply transformation

        cv2.imshow("Processed Image", processed)  # Update only the processed image

    cv2.destroyAllWindows()


def file_selector():
    """
       Opens a file dialog to select an image and processes it interactively.
       """
    root = tk.Tk()
    root.withdraw()  # Hide main Tkinter window
    root.attributes('-topmost', True)

    # Open file dialog for image selection
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files",
                                                                                "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    root.destroy()  # Destroy Tkinter instance

    if file_path:  # If a file is selected, process it interactively
        process_image_interactive(file_path)
    else:
        print("No file selected. Exiting...")
