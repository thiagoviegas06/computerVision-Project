import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import glob
import os
import patternsHelper as ph
from imageHelper import Image as imageHelper


def main(skyfile, patches, pattern):
    # Load sky image
    mensa = imageHelper(skyfile, patches)
    mensa.detect_stars_from_sky(thresh_value=180)
    mensa.iterate_through_patches()
    coords_black = mensa.create_black_image_with_coordinates()

    pat_img = cv.imread(pattern, cv.IMREAD_UNCHANGED)
    #Turn pattern to binary mask
    pat, node_mask, line_mask = ph.extract_pattern_from_image(pat_img, name='Pisces')
    print("Shapes: ", line_mask.shape, coords_black.shape)
    score = cv.matchTemplate(coords_black, line_mask, cv.TM_CCOEFF_NORMED)

    _, best, _, loc = cv.minMaxLoc(score)       
    print("NCC score:", best, "at", loc)
    
    h.image_axis_plotting([(coords_black, "Detected Stars"),
                           (node_mask, "Node Mask"),
                           (line_mask, "Line Mask"),
                           ])

if __name__ == "__main__":
    main(skyfile='train/Pisces/pisces_image.tif', patches='train/Pisces/patches/*.png',
         pattern='patterns/pisces_pattern.png')