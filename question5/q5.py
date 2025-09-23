import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import glob
import os
import patternsHelper as ph
LOWES_RATIO = 0.75


def main():

    # Load sky image

    mensa = cv.imread('train/Mensa/mensa_image.tif')

    #Equalize Mensa image
    # equalized_mensa = h.equalize_image(mensa)

    #convert image to grayscale
    mensa_gray = cv.cvtColor(mensa, cv.COLOR_BGR2GRAY)

    # I found that global thresholding worked better for this image
    mensa_tresh = cv.threshold(mensa_gray, 180, 255, cv.THRESH_BINARY)[1]
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
        #patch_adaptive_thresh = cv.adaptiveThreshold(patch_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        patch_thresh = cv.threshold(patch_gray, 180, 255, cv.THRESH_BINARY)[1]

        # Template matching
        score, coordinates = h.template_matching(mensa_tresh, patch_thresh)
        print(f"Matching score for {filename}: {score}")
        print(f"Coordinates (x, y, width, height) for {filename}: {coordinates}")

        if score > 0.75:
            #cv.circle(updated_mensa, coordinates[0], coordinates[1], (255, 0, 0), -1)
            coordinates_list.append(coordinates[0])


    coordinates = np.array(coordinates_list, dtype=np.int32)

    print(coordinates)

    bound = np.array([255,0,0])

    mask = cv.inRange(updated_mensa, bound, bound)

    updated_image = np.full_like(updated_mensa, 255)
    updated_image[mask > 0 ] = mensa[mask > 0]

    updated_image_gray = cv.cvtColor(updated_image, cv.COLOR_BGR2GRAY)

    #trace line between points
    line_mask = np.zeros_like(updated_image_gray, dtype=np.uint8)  # all zeros
    line_mask = h.draw_lines_between_all_points(line_mask, coordinates, thickness=2)  

    # Ensure it's binary (0/255) in case your helper draws anti-aliased lines
    #_, line_mask = cv.threshold(line_mask, 127, 255, cv.THRESH_BINARY)

    # Distance transform expects foreground=0 (features) and background=1
    # Create a 0/1 mask where background=1, lines=0
    bg_mask = (line_mask == 0).astype(np.uint8)   # 1 on background, 0 on lines
    dt = cv.distanceTransform(bg_mask, cv.DIST_L2, 3).astype(np.float32)

    #Turn pattern to binary mask
    pattern = cv.imread('patterns/mensa_pattern.png', cv.IMREAD_UNCHANGED)
    tpl_f, bw_255 = h.turn_pattern_to_binary_mask(pattern)  # tpl_f is 0/1 float32

    fg = float(tpl_f.sum()) + 1e-6          # number of foreground pixels
    res = cv.matchTemplate(-dt, tpl_f, cv.TM_CCORR)  # maximize correlation == minimize sum(DT under ones)
    _, maxVal, _, maxLoc = cv.minMaxLoc(res)

    sum_dist  = -maxVal
    mean_dist = sum_dist / fg
    score     = 1.0 / (1.0 + mean_dist)

    print(f"mean_dist: {mean_dist:.3f}  score: {score:.3f}  loc: {maxLoc}")

    #template match original image
    score = cv.matchTemplate(line_mask, bw_255, cv.TM_CCORR_NORMED)
    _, best, _, loc = cv.minMaxLoc(score)        # higher is better (0..1]
    print("NCC score:", best, "at", loc)

    pat, node_mask, line_mask = ph.extract_pattern_from_image(pattern, name="Mensa")

    print("Nodes:", len(pat.nodes))
    for lbl, node in pat.nodes.items():
        print(lbl, node.position, "deg:", len(node.links))

    print("Edges:", pat.edges)  # list of (a,b)

    # Plotting results
    h.image_axis_plotting([
        (mensa, "Original Mensa Image"),
        #(updated_mensa, "Detected Patches on Mensa Image"),
        (bw_255, "Patch Template Mask"),
        (dt, "Distance Transform of Edges"),
        (line_mask, "Line Mask"),
    ])
    '''
    h.image_axis_plotting([
        (mensa, "Original Mensa Image"),
        #(updated_mensa, "Detected Patches on Mensa Image"),
        (mensa_tresh, "Global Thresholding on Mensa Image"),
    ])
    '''


if __name__ == "__main__":
    main()