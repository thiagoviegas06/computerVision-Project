import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

def equalize_image(image):
    # First, converting image to yCbCr
    image_yCbCr = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)

    # Apply Histogram Equalization
    image_yCbCr[:,:,0] = cv.equalizeHist(image_yCbCr[:,:,0])

    # Convert back to BGR
    equalized_image = cv.cvtColor(image_yCbCr, cv.COLOR_YCrCb2RGB)

    return equalized_image

def image_axis_plotting(image_title_array):
    array_size = len(image_title_array)
    cols = min(3, array_size)   # no more than 3 columns, but shrink if fewer images
    rows = (array_size + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(16, 8))

    # axs could be 2D or 1D or a single axis, so flatten safely
    axs = np.array(axs).reshape(-1)

    for ax, (img, title) in zip(axs, image_title_array):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    # hide unused subplots if any
    for ax in axs[len(image_title_array):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def countStars(binary_image):
    # Count connected components (stars) in the binary image
    num_labels, _ = cv.connectedComponents(binary_image)

    return num_labels - 1

def newCountStars(binary_image):
    # Find contours in the binary image
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return len(contours)

#template matching
def template_matching(image, template):
    
    result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(result)

    h, w = template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    coordinates = {top_left[0], top_left[1], w, h}

    matched_image = image.copy()
    cv.rectangle(matched_image, top_left, bottom_right, (255, 0, 0), 2)

    return matched_image, max_val, coordinates
