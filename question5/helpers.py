import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def equalize_image(image):
    # First, converting image to yCbCr
    image_yCbCr = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)

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

def draw_lines_between_points(image, pointList, color=(0, 255, 0), thickness=2):
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

def assign_with_radius(Ta, P, eps):
    """One-to-one assignment Ta->P, forbidding pairs beyond eps."""
    D = cdist(Ta, P)
    BIG = 1e6
    D[D > eps] = BIG
    r, c = linear_sum_assignment(D)
    ok = D[r, c] < BIG
    return [(int(i), int(j)) for i, j in zip(r[ok], c[ok])]

def fit_scale_translation_ls(Tm, Pm):
    """Least-squares similarity fit (scale + translation, no rotation)."""
    muT = Tm.mean(axis=0); muP = Pm.mean(axis=0)
    T0 = Tm - muT; P0 = Pm - muP
    s = (T0*P0).sum() / (T0*T0).sum()
    b = muP - s*muT
    return s, b

def align_points(T, P, eps_px=6.0, allow_scale=True):
    """
    Align template T (Mx2) to detections P (Nx2).
    Returns: (s, b, matches, accuracy)
      - s: scale (1.0 if allow_scale=False)
      - b: translation (2,)
      - matches: list of (i_in_T, j_in_P)
      - accuracy: len(matches)/len(T)
    """
    T = np.asarray(T, float);  P = np.asarray(P, float)
    if T.size == 0 or P.size == 0:
        return 1.0, np.array([0.,0.]), [], 0.0

    # --- Initial guess: scale by spread, then shift by centroids
    if allow_scale:
        s0 = rms_size(P) / (rms_size(T) + 1e-12)
    else:
        s0 = 1.0
    b0 = P.mean(axis=0) - s0*T.mean(axis=0)
    T0 = s0*T + b0

    # --- First assignment
    matches = assign_with_radius(T0, P, eps_px)
    if not matches:
        return s0, b0, [], 0.0

    # --- Refine on matched pairs
    Tm = T[[i for i,_ in matches]]
    Pm = P[[j for _,j in matches]]
    if allow_scale:
        s, b = fit_scale_translation_ls(Tm, Pm)
    else:
        s, b = 1.0, Pm.mean(axis=0) - Tm.mean(axis=0)

    # --- Final assignment & score
    T1 = s*T + b
    matches = assign_with_radius(T1, P, eps_px)
    acc = len(matches) / len(T)
    return s, b, matches, acc





