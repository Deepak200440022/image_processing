import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ========== Functions for Graphs ==========
def show_rgb_histogram():
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb)
    ax.clear()
    ax.set_title("RGB Color Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Number of Pixels")
    ax.plot(cv2.calcHist([r], [0], None, [256], [0, 256]), color='red', label='Red')
    ax.plot(cv2.calcHist([g], [0], None, [256], [0, 256]), color='green', label='Green')
    ax.plot(cv2.calcHist([b], [0], None, [256], [0, 256]), color='blue', label='Blue')
    ax.set_xlim([0, 256])
    ax.legend()
    ax.grid(True)
    fig.canvas.draw_idle()

def show_grayscale_histogram():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ax.clear()
    ax.set_title("Grayscale Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Number of Pixels")
    ax.hist(gray.ravel(), bins=256, color='gray')
    ax.set_xlim([0, 256])
    ax.grid(True)
    fig.canvas.draw_idle()

def show_hsv_histogram():
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ax.clear()
    ax.set_title("HSV Histograms")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Number of Pixels")
    ax.plot(cv2.calcHist([h], [0], None, [256], [0, 256]), color='purple', label='Hue')
    ax.plot(cv2.calcHist([s], [0], None, [256], [0, 256]), color='orange', label='Saturation')
    ax.plot(cv2.calcHist([v], [0], None, [256], [0, 256]), color='gray', label='Value')
    ax.set_xlim([0, 256])
    ax.legend()
    ax.grid(True)
    fig.canvas.draw_idle()

def show_rgb_pie_chart(event=None):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_sum = np.sum(rgb[:, :, 0])
    g_sum = np.sum(rgb[:, :, 1])
    b_sum = np.sum(rgb[:, :, 2])
    total = r_sum + g_sum + b_sum
    sizes = [(r_sum / total) * 100, (g_sum / total) * 100, (b_sum / total) * 100]
    labels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    ax.clear()
    ax.set_title("RGB Color Percentage")
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    fig.canvas.draw_idle()

def show_brightness_map():
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    ax.clear()
    im = ax.imshow(v, cmap='inferno')
    ax.set_title("Brightness Map (Value Channel)")
    ax.axis('off')
    fig.colorbar(im, ax=ax)
    fig.canvas.draw_idle()

# ========== Load Image ==========
image_path = "samples/flowers.jpg"  # Replace with your path
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image. Please check the path.")
    exit()

print(f"Image shape: {image.shape}")
print(f"Image dimensions: {image.ndim}")
print(image)

# ========== GUI with Buttons ==========
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

ax1 = plt.axes([0.1, 0.15, 0.15, 0.075])
ax2 = plt.axes([0.3, 0.15, 0.15, 0.075])
ax3 = plt.axes([0.5, 0.15, 0.15, 0.075])
ax4 = plt.axes([0.7, 0.15, 0.15, 0.075])
ax5 = plt.axes([0.4, 0.03, 0.2, 0.075])

b1 = Button(ax1, 'RGB Hist')
b2 = Button(ax2, 'Gray Hist')
b3 = Button(ax3, 'HSV Hist')
b4 = Button(ax4, 'RGB Pie')
b5 = Button(ax5, 'Brightness')

b1.on_clicked(lambda event: show_rgb_histogram())
b2.on_clicked(lambda event: show_grayscale_histogram())
b3.on_clicked(lambda event: show_hsv_histogram())
b4.on_clicked(show_rgb_pie_chart)
b5.on_clicked(lambda event: show_brightness_map())

plt.show()