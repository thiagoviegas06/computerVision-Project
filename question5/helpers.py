import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from patternsHelper import Node, Pattern

def equalize_image(image):
    # First, converting image to yCbCr
    image_yCbCr = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

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
        if img.ndim == 2:  # grayscale (2D array)
            ax.imshow(img, cmap='gray')
        else:  # color (3D array, e.g. RGB)
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

    # Draw a white circle at the match location
    center = (top_left[0] + (bottom_right[0] - top_left[0]) // 2,
            top_left[1] + (bottom_right[1] - top_left[1]) // 2)

    radius = max(bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]) // 2

    coordinates = (center, radius)

    return max_val, coordinates

def draw_lines_between_points(image, pointList, color=(255, 255, 255), thickness=2):
    # Ensure it's an ndarray
    pointList = np.array(pointList)

    if len(pointList) == 0:   
        return image

    pointsCovered = []
    current_idx = 0

    while True:
        currentPoint = pointList[current_idx]
        pointsCovered.append(current_idx)

        nearestIdx = None
        minDist = float("inf")

        for idx, point in enumerate(pointList):
            if idx in pointsCovered:
                continue
            d = np.linalg.norm(point - currentPoint)
            if d < minDist:
                minDist = d
                nearestIdx = idx

        if nearestIdx is not None:
            cv.line(image,
                    tuple(currentPoint.astype(int)),
                    tuple(pointList[nearestIdx].astype(int)),
                    color, thickness)
            current_idx = nearestIdx
        else:
            break

    return image

def draw_lines_between_all_points(image, pointList, color=(255, 255, 255), thickness=2):
    # Ensure it's an ndarray
    pointList = np.array(pointList)

    if len(pointList) == 0:   
        return image

    for i in range(len(pointList) - 1):
        for j in range(i + 1, len(pointList)):
            print(f"Drawing line between point {i} and point {j}")
            cv.line(image,
                    tuple(pointList[i].astype(int)),
                    tuple(pointList[j].astype(int)),
                    color, thickness)

    return image

def chamfer_shift(binary_img, T, search_radius=30):
    """
    binary_img: uint8, stars=255, background=0
    T: (M,2) template points in (x,y)
    search_radius: pixels around the centroid shift to search
    returns: best shift (dx, dy), best score
    """
    # Distance transform on inverted mask (0 where stars are)
    inv = (binary_img == 0).astype(np.uint8) * 255
    dist = cv.distanceTransform(inv, cv.DIST_L2, 3)

    # crude initial shift by centroids
    # if you have detected star points P, use: b0 = P.mean(0) - T.mean(0)
    # if not, just start at (0,0)
    b0 = np.array([0.0, 0.0])

    h, w = dist.shape
    T = np.asarray(T, float)

    def score(b):
        pts = np.rint(T + b).astype(int)
        # keep points inside image
        ok = (pts[:,0] >= 0) & (pts[:,0] < w) & (pts[:,1] >= 0) & (pts[:,1] < h)
        if not np.any(ok): 
            return np.inf
        pts = pts[ok]
        return dist[pts[:,1], pts[:,0]].sum()

    best_s = np.inf
    best_b = None
    R = range(-search_radius, search_radius+1)
    for dx in R:
        for dy in R:
            b = b0 + np.array([dx, dy])
            s = score(b)
            if s < best_s:
                best_s, best_b = s, b
    return tuple(best_b), float(best_s)

def rms_size(X):
    """RMS spread (scale proxy) of a 2D point set."""
    X = np.asarray(X, float)
    mu = X.mean(axis=0, keepdims=True)
    return np.sqrt(((X - mu)**2).sum(axis=1).mean())


# Display only n brightest stars in mensa_gray
def get_n_brightest_stars(original_image, processed_gray_image, n):
    """
    Finds and displays the n brightest stars from an image.
    """

    # Find contours in the thresholded image
    contours, _ = cv.findContours(processed_gray_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if not contours:
        print("No bright spots (contours) found in the image.")
        return

    # 5. Sort contours by area in descending order
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    # 6. Create a new, black image to draw the selected stars
    mask = np.zeros_like(original_image)

    # 7. Draw the top 'n' contours as white on the black mask
    for i in range(min(n, len(contours))):
        cv.drawContours(mask, [contours[i]], -1, (255, 255, 255), -1)

    # 8. Use a bitwise AND operation to display only the top n stars from the original image
    brightest_stars_only = cv.bitwise_and(original_image, mask)


    return cv.cvtColor(brightest_stars_only, cv.COLOR_BGR2GRAY)


def turn_pattern_to_binary_mask(pattern):
    assert pattern is not None and pattern.ndim == 3

    if pattern.shape[2] == 4:
        bgr   = pattern[:, :, :3]
        alpha = pattern[:, :, 3]
        gray  = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

        # Keep pixels that are somewhat opaque AND not near-black
        fg = (alpha > 10) & (gray > 15)
        tpl_bin = np.where(fg, 255, 0).astype(np.uint8)
    else:
        hsv = cv.cvtColor(pattern, cv.COLOR_BGR2HSV)
        # White dots: high V, low S (broaden a bit)
        white = cv.inRange(hsv, (0, 0, 180), (180, 50, 255))
        # Green lines: widen hue range slightly
        green = cv.inRange(hsv, (30, 30, 40), (90, 255, 255))
        tpl_bin = cv.bitwise_or(white, green)

    # Clean & thickness adjust (optional but helpful)
    tpl_bin = cv.morphologyEx(tpl_bin, cv.MORPH_OPEN,  np.ones((2,2), np.uint8))
    tpl_bin = cv.morphologyEx(tpl_bin, cv.MORPH_DILATE, np.ones((2,2), np.uint8))

    tpl_f = (tpl_bin > 0).astype(np.float32)  # 0/1 float32 for matchTemplate with -dt
    return tpl_f, tpl_bin


import math, numpy as np
from collections import defaultdict

def _wrap(a):  # [-pi, pi)
    return (a + math.pi) % (2*math.pi) - math.pi

def test_for_match(node,
                   coordinates_list,
                   angle_tolerance_deg=12.0,
                   len_tolerance_ratio=0.20,
                   pos_tol_px=6.0,
                   min_votes=2):
    """
    node.links: Dict[str, (ux, uy, L, ang)]
    coordinates_list: list[(x,y)] scene points
    Returns: dict with theta, scale, anchor_idx, t (translation), votes, M (2x3) or None
    """
    pts = np.asarray(coordinates_list, dtype=np.float32)
    n = len(pts)
    if n < 2 or len(node.links) == 0:
        return None

    ang_tol = math.radians(angle_tolerance_deg)

    # 1) Hypotheses (theta, scale) from all template links vs all scene pairs
    hyps = []
    for _, (_, _, L, ang_tpl) in node.links.items():
        if L <= 1e-6:
            continue
        for i in range(n):
            for j in range(i+1, n):
                dx = float(pts[j,0] - pts[i,0])
                dy = float(pts[j,1] - pts[i,1])
                local_len = math.hypot(dx, dy)
                if local_len <= 1e-6:
                    continue
                local_ang = math.atan2(dy, dx)
                # two directions (undirected edge)
                for a in (local_ang, _wrap(local_ang + math.pi)):
                    theta = _wrap(a - ang_tpl)
                    s = local_len / L
                    hyps.append((theta, s))

    if not hyps:
        return None
    hyps = np.asarray(hyps, np.float32)

    # 2) Coarse binning (Hough-like vote) to find a consistent (theta, s)
    theta_bin = max(ang_tol, math.radians(10))  # ~10-12°
    s_bin = max(1e-6, len_tolerance_ratio * 0.10)  # ~10% bins
    theta_norm = (hyps[:,0] + math.pi) % (2*math.pi)
    t_idx = np.floor(theta_norm / theta_bin).astype(int)
    s_idx = np.floor(hyps[:,1] / s_bin).astype(int)

    buckets = defaultdict(list)
    for k, key in enumerate(zip(t_idx, s_idx)):
        buckets[key].append(k)

    best_key, best_idxs = max(buckets.items(), key=lambda kv: len(kv[1]))
    if len(best_idxs) < min_votes:
        return None

    theta_est = float(_wrap(hyps[best_idxs,0].mean()))
    scale_est = float(np.median(hyps[best_idxs,1]))

    # 3) Choose the best scene anchor & compute translation
    # For each scene point as anchor, check how many links land near some scene point.
    c, s = math.cos(theta_est), math.sin(theta_est)
    R = np.array([[c, -s],[s, c]], dtype=np.float32) * scale_est

    def votes_for_anchor(anchor_idx):
        anchor = pts[anchor_idx]
        votes = 0
        for _, (ux, uy, L, ang_tpl) in node.links.items():
            pred = anchor + (R @ np.array([ux*L, uy*L], np.float32))
            # nearest neighbor test
            d2 = ((pts - pred)**2).sum(axis=1)
            if float(d2.min()) <= pos_tol_px**2:
                votes += 1
        return votes

    anchor_votes = [votes_for_anchor(i) for i in range(n)]
    best_anchor = int(np.argmax(anchor_votes))
    votes = anchor_votes[best_anchor]
    if votes < min_votes:
        return None

    # 4) Build full 2x3 similarity matrix M = [R | t]
    # Map template anchor node position to the chosen scene anchor
    p_anchor = np.array(node.position, dtype=np.float32)  # template coords
    t = pts[best_anchor] - (R @ p_anchor)
    M = np.zeros((2,3), np.float32)
    M[:,:2] = R
    M[:, 2] = t

    return {
        "theta": theta_est,
        "scale": scale_est,
        "anchor_idx": best_anchor,
        "t": (float(t[0]), float(t[1])),
        "votes": votes,
        "M": M
    }

def calculate_coordinate_angles(coordinates):
    """
    Given a list of (x, y) coordinates, calculate the angle of each point
    relative to all other points in the list.
    Angles are measured in radians from the positive x-axis.
    """
    if len(coordinates) == 0:
        return []

    pts = np.array(coordinates, dtype=np.float32)
    n = len(pts)
    angles = []
    for i in range(n):
        point_angles = []
        for j in range(n):
            if i != j:
                dx = float(pts[j, 0] - pts[i, 0])
                dy = float(pts[j, 1] - pts[i, 1])
                angle = math.atan2(dy, dx)  # Angle in radians
                point_angles.append(angle)
        angles.append(point_angles)
    return angles

def compareAngles(angle1, angle2, tolerance_deg=12.0):
    """
    Compare two angles (in radians) and determine if they are within a specified tolerance.
    The comparison accounts for the circular nature of angles.
    """
    tolerance_rad = math.radians(tolerance_deg)
    diff = abs(angle1 - angle2) % (2 * math.pi)
    return diff <= tolerance_rad or diff >= (2 * math.pi - tolerance_rad)