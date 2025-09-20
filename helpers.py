import cv2 as cv
import numpy as np

def luminanceDifferenceForGreen(image):
    b, g, r = cv.split(image)  
    g = g.astype(np.float32)
    r = r.astype(np.float32)
    b = b.astype(np.float32)

    # Reconstruct approximate green channel
    new_g = (g - 0.3 * r - 0.11 * b) / 0.59

    # Clip to valid range [0,255]
    new_g = np.clip(new_g, 0, 255).astype(np.uint8)

    # Rebuild the corrected image
    lum = cv.merge((b.astype(np.uint8), new_g, r.astype(np.uint8)))
    return lum

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def histogram_equalization(image):
    # Convert to YCrCb color space
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    # Equalize the Y channel
    ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
    # Convert back to BGR color space
    equalized_image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)
    return equalized_image


def treshholding(image, tresh = 150):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, tresh, 255, cv.THRESH_BINARY)
    return binary

def adaptive_treshholding(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv.THRESH_BINARY, 11, 2)
    return binary

def countStars(binary_image):
    # Count connected components (stars) in the binary image
    num_labels, _ = cv.connectedComponents(binary_image)

    return num_labels - 1

def newCountStars(binary_image):
    # Find contours in the binary image
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return len(contours)

def calculate_histogram(image):
    rgb_hist = []
    colors = ('r','g','b')
    for i, color in enumerate(colors):
        rgb_hist.append(cv.calcHist([image], [i], None, [256], [0, 256]))

    return rgb_hist

def generatePlot(rows, columns, images):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, columns, figsize=(18, 6))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        if len(img.shape) == 2:  # Grayscale image
            ax.imshow(img, cmap='gray')
        else:  # Color image
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.axis('off')

    plt.tight_layout()
    plt.show()