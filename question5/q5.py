import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import glob
import os

# Load sky image

mensa = cv.imread('train/Mensa/mensa_image.tif')

#Equalize Mensa image
equalized_mensa = h.equalize_image(mensa)

#convert image to grayscale
mensa_gray = cv.cvtColor(mensa, cv.COLOR_BGR2GRAY)

#adaptive thresholding
mensa_adaptive_thresh = cv.adaptiveThreshold(mensa_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

updated_mensa = mensa.copy()

coordinates_list = []

#iterate through each patch and try to template matching
path_to_patches = 'train/Mensa/patches/*.png'
for patch_file in glob.glob(path_to_patches):
    filename = os.path.basename(patch_file)
    print(f"Processing patch: {filename}")

    # Load patch image
    patch = cv.imread(patch_file)

    # Convert patch to grayscale
    patch_gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to the patch
    patch_adaptive_thresh = cv.adaptiveThreshold(patch_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Template matching
    score, coordinates = h.template_matching(mensa_adaptive_thresh, patch_adaptive_thresh)
    print(f"Matching score for {filename}: {score}")
    print(f"Coordinates (x, y, width, height) for {filename}: {coordinates}")

    if score > 0.75:
        cv.circle(updated_mensa, coordinates[0], coordinates[1], (255, 0, 0), -1)
        coordinates_list.append(coordinates[0])


coordinates = np.array(coordinates_list, dtype=np.int32)

print(coordinates)

#convert to hsv
mensa_hsv = cv.cvtColor(mensa, cv.COLOR_BGR2HSV)

bound = np.array([255,0,0])

mask = cv.inRange(updated_mensa, bound, bound)

updated_image = np.full_like(updated_mensa, 255)
updated_image[mask > 0 ] = mensa[mask > 0]

updated_image_gray = cv.cvtColor(updated_image, cv.COLOR_BGR2GRAY)

#trace line between points
#image_with_lines = h.draw_lines_between_points(updated_mensa, coordinates, thickness=2)



#Try to template match pattern on line image

pattern = cv.imread('patterns/mensa_pattern.png', cv.IMREAD_GRAYSCALE)

# Tune 't' (try 180–220 depending on your image)
t = 200
_, bw = cv.threshold(pattern, t, 255, cv.THRESH_BINARY)

# 2) Optional denoise small specks
bw = cv.medianBlur(bw, 3)

# 3) Connected components + shape filtering (keep round blobs)
num, labels, stats, centroids = cv.connectedComponentsWithStats(bw)

T = centroids[1:].astype(np.float32)

print("Template points:", T.shape)
print(T)


inverted = cv.bitwise_not(updated_image_gray)

wrong_pattern = cv.imread('patterns/indus_pattern.png', cv.IMREAD_GRAYSCALE)
# Plotting results
h.image_axis_plotting([
    (mensa, "Original Mensa Image"),
    (mensa_adaptive_thresh, "Adaptive Thresholding on Mensa Image"),
    (updated_mensa, "Detected Patches on Mensa Image"),
    (inverted, "Inverted Gray Scale Tresholded Image"),
    (pattern, "Correct Pattern Image"),
    (bw, "Binarized Correct Pattern"),
    (wrong_pattern, "Wrong Pattern Image")
])

#template matching 
score, coordinates = h.template_matching(inverted, bw)
print(f"Final Matching score for pattern: {score}")
print(f"Final Coordinates (x, y, width, height) for pattern: {coordinates}")

wrong_score, wrong_coordinates = h.template_matching(inverted, wrong_pattern)

