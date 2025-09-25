import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import os
import glob
from patternsHelper import Node, Pattern, extract_pattern_from_image
import math
import helpers as h

class Image:
    def __init__(self, image_path, path_to_patches=None):
        self.image = cv.imread(image_path)
        print(f"Loaded image from {image_path} with shape {self.image.shape if self.image is not None else 'None'}")
        self.path_to_patches = path_to_patches
        self.coordinates_list = []
        self.global_thresh_image = None
        self.patches = {}
        self.constelation = None
        self.confindence_score = -1
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
        num = 1
        for patch_file in sorted(glob.glob(self.path_to_patches)):
            filename = os.path.basename(patch_file)
            print(f"Processing patch: {filename}")
            keyValue = f"patch {num}"

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

            if score > 0.95:
                self.coordinates_list.append(coordinates[0])

                self.patches[keyValue] = coordinates[0]
                print(f"{keyValue} matched at {coordinates[0]}")

            else:
                self.patches[keyValue] = - 1
                print(f"{keyValue} matched at -1 (no match)")
            num += 1
    
    def get_coordinates(self):
        return self.coordinates_list
    
    def detect_stars_from_sky(self, thresh_value=180, min_area=4, max_area=1000):
        """
        Detects all stars directly from the sky image using contour detection.
        This is more robust than iterating through patches.
        
        Returns: A list of tuples, where each is ((x, y), size).
        """
        if self.global_thresh_image is None:
            self.global_threshold(thresh_value)
        
        contours, _ = cv.findContours(self.global_thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        detected_stars = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            
            if min_area <= area <= max_area:
                # Calculate the centroid (center of mass) for a precise position
                M = cv.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detected_stars.append( ((cx, cy), area) )
        
        self.detected_stars = detected_stars
        return self.detected_stars

    def create_black_image_with_coordinates(self):
        # Make a 2D black canvas, uint8, single channel
        H, W = self.image.shape[:2]
        black_image = np.zeros((H, W), dtype=np.uint8)

        # Now draw lines in white (255)
        updated_image = h.draw_lines_between_all_points(
            black_image, self.coordinates_list, color=255  # just an int, not a tuple
        )
        return updated_image
    
    def get_patch_dict(self):
        return self.patches
    
    def get_constelation(self):
        return self.constelation
    
    def set_constelation(self, constelation):
        self.constelation = constelation

    def get_confidence_score(self):
        return self.confindence_score
    
    def set_confidence_score(self, score):
        self.confindence_score = score
