import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import os
import glob
from patternsHelper import Node, Pattern, extract_pattern_from_image
import math

class Image:
    def __init__(self, image_path, path_to_patches=None):
        self.image = cv.imread(image_path)
        self.path_to_patches = path_to_patches
        self.coordinates_list = []
        self.global_thresh_image = None
        if self.image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        self.gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    def equalize(self):
        self.equalized_image = h.equalize_image(self.image)
        return self.equalized_image

    def adaptive_threshold(self, block_size=11, C=2):
        if block_size % 2 == 0:
            raise ValueError("Block size must be an odd number.")
        self.adaptive_thresh_image = cv.adaptiveThreshold(
            self.gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, block_size, C)
        return self.adaptive_thresh_image

    def global_threshold(self, thresh_value=180):
        _, self.global_thresh_image = cv.threshold(
            self.gray_image, thresh_value, 255, cv.THRESH_BINARY)
        return self.global_thresh_image
    
    def iterate_through_patches(self):
        for patch_file in glob.glob(self.path_to_patches):
            filename = os.path.basename(patch_file)
            print(f"Processing patch: {filename}")

            # Load patch image
            patch = cv.imread(patch_file)

            if patch is None:
                print(f"Patch at path {patch_file} could not be loaded.")
                continue

            # Convert patch to grayscale
            patch_gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
            
            # Apply global thresholding to the patch
            patch_thresh = cv.threshold(patch_gray, 180, 255, cv.THRESH_BINARY)[1]

            # Template matching
            score, coordinates = h.template_matching(self.global_thresh_image, patch_thresh)
            print(f"Matching score for {filename}: {score}")
            print(f"Coordinates (x, y, width, height) for {filename}: {coordinates}")

            if score > 0.75:
               self.coordinates_list.append(coordinates[0])
    
    def get_coordinates(self):
        return self.coordinates_list
