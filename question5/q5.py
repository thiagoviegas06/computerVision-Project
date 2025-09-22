import cv2 as cv
#import numpy as np
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
    matched_image, score, coordinates = h.template_matching(mensa_adaptive_thresh, patch_adaptive_thresh)
    print(f"Matching score for {filename}: {score}")
    print(f"Coordinates (x, y, width, height) for {filename}: {coordinates}")




