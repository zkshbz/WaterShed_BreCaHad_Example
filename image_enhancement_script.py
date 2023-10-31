import cv2
import numpy as np

def histogram_equlization(source_image):
    # Read the image
    image = cv2.imread(source_image, cv2.IMREAD_COLOR)

    # Split the image into its RGB channels
    b, g, r = cv2.split(image)

    # Equalize the histogram of each channel
    equalized_b = cv2.equalizeHist(b)
    equalized_g = cv2.equalizeHist(g)
    equalized_r = cv2.equalizeHist(r)

    # Merge the equalized channels back into a color image
    equalized_image = cv2.merge((equalized_b, equalized_g, equalized_r))
    
    # Save or display the equalized image
    cv2.imwrite('equalized_image.jpg', equalized_image)


def constrat_strecting(source_image):
    # Read the color image
    image = cv2.imread(source_image, cv2.IMREAD_COLOR)

    # Split the image into its RGB channels
    b, g, r = cv2.split(image)

    # Perform stretching for each channel
    stretched_b = np.uint8(255 * (b - np.min(b)) / (np.max(b) - np.min(b)))
    stretched_g = np.uint8(255 * (g - np.min(g)) / (np.max(g) - np.min(g)))
    stretched_r = np.uint8(255 * (r - np.min(r)) / (np.max(r) - np.min(r)))

    # Merge the stretched channels back into a color image
    stretched_image = cv2.merge((stretched_b, stretched_g, stretched_r))

    # Save or display the stretched image
    cv2.imwrite('stretched_image.jpg', stretched_image)