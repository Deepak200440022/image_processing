import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Capture two frames from the webcam
cap = cv2.VideoCapture(0)
print("Capturing frame 1...")
ret1, frame1 = cap.read()

# Delay of 200 milliseconds
cv2.waitKey(200)

print("Capturing frame 2...")
ret2, frame2 = cap.read()
cap.release()

if not ret1 or not ret2:
    print("Failed to capture frames.")
    exit()

# Step 2: Convert both frames to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Step 3: Calculate Optical Flow (Farneback method)
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2,
    None,
    0.5,  # pyramid scale
    3,    # pyramid levels
    15,   # window size
    3,    # iterations
    5,    # poly_n
    1.2,  # poly_sigma
    0     # flags
)

# Step 4: Visualize using arrows (quiver plot)
h, w = gray1.shape
step = 16
y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int32)

fx, fy = flow[y, x].T

plt.figure(figsize=(10, 6))
plt.imshow(gray1, cmap='gray')
plt.quiver(x, y, fx, fy, color='red', angles='xy', scale_units='xy', scale=1)
plt.title("Optical Flow - Motion Arrows")
plt.axis('off')
plt.show()
