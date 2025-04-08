import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_IO import grayscale_luminosity
# Function to display RGB histogram
def show_rgb_histogram(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb)

    # Plot the histogram
    plt.figure()
    plt.title("RGB Color Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.plot(cv2.calcHist([r], [0], None, [256], [0, 256]), color='red', label='Red')
    plt.plot(cv2.calcHist([g], [0], None, [256], [0, 256]), color='green', label='Green')
    plt.plot(cv2.calcHist([b], [0], None, [256], [0, 256]), color='blue', label='Blue')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True)
    plt.show()
def show_gray_histogram(gray_img):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.hist(gray_img.ravel(), bins=256, color='gray')
    plt.xlim([0, 256])
    plt.show()
# === Main Code ===

# Load image from location (change this path)
image_path = "samples/flower.jpg"  # Replace with your actual image path
image = cv2.imread(image_path)

def show_rgb_pie(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_sum = np.sum(rgb[:, :, 0])
    g_sum = np.sum(rgb[:, :, 1])
    b_sum = np.sum(rgb[:, :, 2])
    total = r_sum + g_sum + b_sum
    labels = ['Red', 'Green', 'Blue']
    sizes = [r_sum/total*100, g_sum/total*100, b_sum/total*100]
    colors = ['red', 'green', 'blue']

    plt.figure()
    plt.title("RGB Percentage Pie Chart")
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.axis('equal')
    plt.show()

# Check if image loaded
if image is None:
    print("Failed to load image. Check the path.")
else:
    # Print shape and dimensions
    print(f"Image shape: {image.shape}")        # (height, width, channels)
    print(f"Image dimensions: {image.ndim}")    # Typically 3 for color images
    # show_gray_histogram(grayscale_luminosity(image))
    show_rgb_pie(image)
    # Show RGB histogram
    # show_rgb_histogram(image)

