import numpy as np  # our main objective
import cv2 # opening and showing images from the system
import os

image_path =  r"C:\Users\HP\Downloads\sample_image.jpg"

def open_image(path):
    """checks if the image exists or not , if exists returns the image"""
    if os.path.exists(path):
        img = cv2.imread(image_path)
        return img



img = open_image(image_path)
print(img)