import numpy as np  # our main objective
import cv2 # opening and showing images from the system
import os


#change the path of the image to your image
image_path =  r"C:\Users\HP\Downloads\sample_image.jpg"

def open_image(path):
    """checks if the image exists or not , if exists returns the image"""
    if os.path.exists(path):
        img = cv2.imread(image_path)
        return img

def show_image(image):
    """gets the image and show it"""
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 1 -> Rotation of the image
def rotate_90(img, clockwise = True):
    #(H, W, 3) (Height × Width × Channels)

    if(clockwise):
        return np.transpose(img, (1, 0, 2))[:, ::-1]
    else:
        return np.transpose(img,(1,0,2))

def rotate_180(img):
    return img[::-1,::-1]


def rotate_degree(image,angle):
    #change the angle to radian
    theta = np.radians(angle)



    #find the image center
    height,width,colors = image.shape
    center_height , center_width = height//2 , width //2

    #the rotation matrix
    # |cos(b)  -sin(b)|
    # |sin(b)   cos(b)|
    cos_b, sin_b = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_b,-sin_b],
                                [sin_b,cos_b]])

     # corners
    # generaly the image's top-left corner is at 0,0 so we moves the image to get it to the center
    # so that the center is at 0,0  so the new coordinates of the corners will be
    corners = np.array([[-center_width,- center_height], #top-left cornor
          [center_width, -center_height],# top-right cornor
          [-center_width,center_height], # bottom-left cornor
          [center_width,center_height] # bottom-right cornor
          ])
    # new corners
    new_corners = np.dot(rotation_matrix, corners.T).T



    # now the image is rotated (dimand roated square)but the output still should be a square so we create new box to enclose the rotated image
    # finding maximum and minimum corners
    min_x, min_y = new_corners.min(axis=0)
    max_x, max_y = new_corners.max(axis=0)
    # calculating the new width and height
    new_width = int(max_x - min_x)
    new_height = int(max_y - min_y)

    # create a new image
    rotated_image = np.zeros((new_height,new_width,colors),dtype=np.uint8) # all zero that is the whole image is black right now

    #change the color of the image
    # mapping the rotated pixels to the new image
    for y in range(new_height):
        for x in range(new_width):
            #find the original coordinates of the image
            #rotatingt the matrix to map
            orig_x , orig_y = np.dot(rotation_matrix.T, [x + min_x, y + min_y])
            orig_x , orig_y = int(orig_x + center_width) , int(orig_y + center_height)

            #using bilinear Interpolation
            # finds the surrounding pixles
            x1,y1 = int(np.floor(orig_x)),int(np.floor(orig_y))
            x2, y2 = x1 + 1, y1 + 1

            if 0 <= x1 < width - 1 and 0 <= y1 < height - 1:
                a = orig_x - x1  # Fractional part of x
                b = orig_y - y1  # Fractional part of y

            #     # Fetch the four neighboring pixels
                I11 = image[y1, x1]
                I12 = image[y2, x1]
                I21 = image[y1, x2]
                I22 = image[y2, x2]
            #
                # Compute bilinear interpolation
                interpolated_pixel = (
                        I11 * (1 - a) * (1 - b) +
                        I21 * a * (1 - b) +
                        I12 * (1 - a) * b +
                        I22 * a * b
                )
                rotated_image[y, x] = interpolated_pixel.astype(np.uint8)

    return rotated_image



def flipper(img , mode = "horizontal"):
    """
    mode: "horizontal", "vertical", or "both"
    """
    if(mode == "horizontal"):
        return

img = open_image(image_path)
img =rotate_degree(img,-45)
show_image(img)