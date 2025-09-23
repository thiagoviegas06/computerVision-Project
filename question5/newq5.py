import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import glob
import os
import patternsHelper as ph
from imageHelper import Image as imageHelper
import math

LOWES_RATIO = 0.75


def main():

    # Load sky image
    mensa = imageHelper('train/Mensa/mensa_image.tif', path_to_patches='train/Mensa/patches/*.png')
    mensa.global_threshold(thresh_value=180)

    mensa.iterate_through_patches()

    coordinates = mensa.get_coordinates()

    print(coordinates)
    #Turn pattern to binary mask

    angles = h.calculate_coordinate_angles(coordinates)
    print(angles)

    pattern = cv.imread('patterns/mensa_pattern.png', cv.IMREAD_UNCHANGED)
    pat, node_mask, line_mask = ph.extract_pattern_from_image(pattern, name="Mensa")

    print("Nodes:", len(pat.nodes))
    for lbl, node in pat.nodes.items():
        print(lbl, node.position, "deg:", len(node.links))

    print("Edges:", pat.edges)  # list of (a,b)

    highest_deg_node = pat.getHighestDegreeNode()

    print("Anchor node:", highest_deg_node.label, highest_deg_node.position, "deg:", len(highest_deg_node.links))

   


if __name__ == "__main__":
    main()