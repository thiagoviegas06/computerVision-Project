import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import glob
import os
import patternsHelper as ph
from imageHelper import Image as imageHelper


def main(skyfile, patches):
    # Load sky image
    mensa = imageHelper(skyfile, patches)
    mensa.detect_stars_from_sky(thresh_value=180)
    mensa.iterate_through_patches()
    coords_black = mensa.create_black_image_with_coordinates()

    h.image_axis_plotting([(coords_black, "Detected Stars")],)

    pass

if __name__ == "__main__":
    main(skyfile='train/Pisces/pisces_image.tif', patches='train/Pisces/patches/*.png')