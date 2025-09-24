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


def visualize_pattern(node_mask, line_mask, title="Pattern Visualization"):
    plt.figure(figsize=(8, 8))
    plt.imshow(node_mask, cmap='gray')
    plt.imshow(line_mask, cmap='Greens', alpha=0.5)  # Overlay lines with some transparency
    plt.title(title)
    plt.axis('off')
    plt.show()

def calculate_all_angle_signatures(coordinates: list[tuple[float, float]]) -> list[list[float]]:
    """
    For each coordinate, calculates a sorted list of angles to all other coordinates.
    This creates an "angle signature" for each point.

    Args:
        coordinates: A list of (x, y) coordinates from the sky image.

    Returns:
        A list of signatures, where each signature is a sorted list of angles in radians.
    """
    all_signatures = []
    for i, p1 in enumerate(coordinates):
        angles = []
        for j, p2 in enumerate(coordinates):
            if i == j:
                continue
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = math.atan2(dy, dx)
            angles.append(angle)
        angles.sort()
        all_signatures.append(angles)
    return all_signatures

def _normalize_angle(angle):
    """Helper to normalize an angle to the range [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def compare_signatures(pattern_sig: list[float], detected_sig: list[float], tolerance_rad: float) -> tuple[int, float | None]:
    """
    Compares signatures and returns the match count AND the best rotational offset.
    """
    if not pattern_sig or not detected_sig:
        return 0, None

    max_matches = 0
    best_rotation_offset = None # Will store the best rotation found

    for p_angle_anchor in pattern_sig:
        for d_angle_anchor in detected_sig:
            rotation_offset = _normalize_angle(d_angle_anchor - p_angle_anchor)
            
            current_matches = 0
            temp_detected_sig = list(detected_sig)
            
            for p_angle in pattern_sig:
                expected_d_angle = _normalize_angle(p_angle + rotation_offset)
                
                best_match_idx = -1
                min_diff = float('inf')
                
                for i, d_angle_candidate in enumerate(temp_detected_sig):
                    diff = abs(_normalize_angle(d_angle_candidate - expected_d_angle))
                    if diff < min_diff:
                        min_diff = diff
                        best_match_idx = i
                
                if best_match_idx != -1 and min_diff <= tolerance_rad:
                    current_matches += 1
                    temp_detected_sig.pop(best_match_idx)

            if current_matches > max_matches:
                max_matches = current_matches
                best_rotation_offset = rotation_offset # Save the best rotation
                
    return max_matches, best_rotation_offset

def calculate_all_angle_signatures(coordinates: list[tuple[float, float]], k: int = 12) -> list[list[float]]:
    """
    For each coordinate, calculates a sorted list of angles to its 'k' nearest neighbors.
    This creates a sparse, local "angle signature" for each point.

    Args:
        coordinates: A list of (x, y) coordinates from the sky image.
        k (int): The number of nearest neighbors to use for the signature. A value
                 around 10-15 is often robust.

    Returns:
        A list of signatures, where each signature is a sorted list of angles in radians.
    """
    if len(coordinates) <= k:
        # If there are fewer points than k, fall back to the old method (all points)
        # This is a fallback for very sparse images.
        all_signatures = []
        for i, p1 in enumerate(coordinates):
            angles = []
            for j, p2 in enumerate(coordinates):
                if i == j: continue
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                angles.append(math.atan2(dy, dx))
            angles.sort()
            all_signatures.append(angles)
        return all_signatures

    # Use a k-d tree for super-fast nearest neighbor searches
    coords_array = np.array(coordinates)
    kdtree = cKDTree(coords_array)
    
    # Query the tree for the k+1 nearest neighbors for all points at once
    # (k+1 because the point itself is its own nearest neighbor)
    distances, indices = kdtree.query(coords_array, k=k + 1)

    all_signatures = []
    for i in range(len(coordinates)):
        p1 = coords_array[i]
        angles = []
        # Iterate through the k nearest neighbors (skipping the first, which is the point itself)
        for neighbor_idx in indices[i, 1:]:
            p2 = coords_array[neighbor_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angles.append(math.atan2(dy, dx))
        
        angles.sort()
        all_signatures.append(angles)
        
    return all_signatures

from collections import defaultdict

def find_candidate_rotations(pattern_sig: list[float], detected_sig: list[float], tolerance_rad: float, num_candidates: int = 4):
    """
    Finds the top N rotational hypotheses by voting, then verifies each with a true match count.
    """
    if not pattern_sig or not detected_sig:
        return []

    # Stage 1: Fast voting to find promising rotation bins
    rotation_votes = defaultdict(int)
    rotation_offsets = defaultdict(list)
    bin_size_rad = tolerance_rad 
    
    for p_angle_anchor in pattern_sig:
        for d_angle_anchor in detected_sig:
            rotation_offset = _normalize_angle(d_angle_anchor - p_angle_anchor)
            rotation_bin = round(rotation_offset / bin_size_rad)
            rotation_votes[rotation_bin] += 1
            rotation_offsets[rotation_bin].append(rotation_offset)

    if not rotation_votes:
        return []

    # Get the top N most-voted-for rotation bins
    sorted_bins = sorted(rotation_votes.items(), key=lambda item: item[1], reverse=True)
    
    candidates = []
    # Stage 2: For each top candidate, perform a precise 1-to-1 match to get an accurate score
    for i in range(min(num_candidates, len(sorted_bins))):
        best_bin, _ = sorted_bins[i]
        # Use the average rotation in the bin for better accuracy
        avg_rotation = np.mean(rotation_offsets[best_bin])

        # Perform a true match count using this rotation
        temp_detected_sig = list(detected_sig)
        current_matches = 0
        for p_angle in pattern_sig:
            expected_d_angle = _normalize_angle(p_angle + avg_rotation)
            
            # Find the best corresponding angle in the detected signature
            best_match_idx = -1
            min_diff = float('inf')
            
            for k, d_angle_candidate in enumerate(temp_detected_sig):
                diff = abs(_normalize_angle(d_angle_candidate - expected_d_angle))
                if diff < min_diff:
                    min_diff = diff
                    best_match_idx = k
            
            # If a match is found within tolerance, count it and "consume" the detected angle
            if best_match_idx != -1 and min_diff <= tolerance_rad:
                current_matches += 1
                temp_detected_sig.pop(best_match_idx)

        # The score is now a true match count, which cannot exceed len(pattern_sig)
        candidates.append({'rotation_rad': avg_rotation, 'score': current_matches})
        
    return candidates

# You'll also need to update find_best_match_by_angles to use this new function.
# This function is now a simple wrapper.

def find_best_match_by_angles(pattern: Pattern, detected_coords: list, anchor_node: Node, angle_tolerance_deg: float = 10.0, k_neighbors: int = 12):
    """
    Finds multiple candidate matches for a given anchor node.
    """
    if not detected_coords or len(detected_coords) < 2 or not anchor_node or not anchor_node.links:
        return []

    pattern_sig = sorted([link[3] for link in anchor_node.links.values()])
    tolerance_rad = math.radians(angle_tolerance_deg)

    candidate_results = []
    # Iterate through every star in the sky as a potential anchor point
    for i, p_coord in enumerate(detected_coords):
        detected_sig = calculate_all_angle_signatures([p_coord] + detected_coords[:i] + detected_coords[i+1:], k=k_neighbors)[0]
        
        # Get top rotation candidates for this sky point
        rotations = find_candidate_rotations(pattern_sig, detected_sig, tolerance_rad)
        
        for rot_candidate in rotations:
            candidate_results.append({
                "name": pattern.name,
                "score": rot_candidate['score'],
                "normalized_score": rot_candidate['score'] / len(pattern_sig) if pattern_sig else 0.0,
                "anchor_idx": i,
                "rotation_rad": rot_candidate['rotation_rad'],
                "pattern": pattern,
                "pattern_anchor_node": anchor_node
            })

    # Return all found candidates, let the main loop sort them out
    return candidate_results

def display_match(sky_image, detected_coords, match_result):
    """
    Overlays the FULL matched constellation pattern onto the sky image using the
    calculated geometric transformation.
    """
    vis_image = cv.cvtColor(sky_image, cv.COLOR_GRAY2BGR) if len(sky_image.shape) == 2 else sky_image.copy()

    # Draw all detected stars for context
    for x, y in detected_coords:
        cv.circle(vis_image, (int(x), int(y)), 7, (255, 100, 100), 1)

    # Check if a valid match and transform exist
    if match_result.get('final_score', 0.0) < 0.1 or 'transform' not in match_result:
        print("Match score too low or transform not found. Cannot display full overlay.")
        # Highlight the anchor if it exists
        if match_result.get('anchor_idx', -1) != -1:
            anchor_pos = detected_coords[match_result['anchor_idx']]
            cv.circle(vis_image, (int(anchor_pos[0]), int(anchor_pos[1])), 12, (0, 0, 255), 2)
    else:
        # --- A good match was found, so draw the full projection ---
        transform = match_result['transform']
        R, t = transform['R'], transform['t']
        pattern = match_result['pattern']
        
        # Project all pattern nodes into the scene
        all_pattern_nodes_pos = np.array([node.position for node in pattern.nodes.values()])
        projected_nodes = (R @ all_pattern_nodes_pos.T).T + t
        
        # Create a map from node label to its index for drawing edges
        label_to_idx = {label: i for i, label in enumerate(pattern.nodes.keys())}

        # Draw projected edges
        for start_label, end_label in pattern.edges:
            p1 = tuple(projected_nodes[label_to_idx[start_label]].astype(int))
            p2 = tuple(projected_nodes[label_to_idx[end_label]].astype(int))
            cv.line(vis_image, p1, p2, (0, 255, 0), 2) # Green lines
        
        # Draw projected nodes, color-coded by whether they matched a real star
        scene_kdtree = cKDTree(np.array(detected_coords))
        distances, _ = scene_kdtree.query(projected_nodes)
        
        for i, pos in enumerate(projected_nodes):
            # If the projected node is close to a real star, it's a confirmed match
            if distances[i] < 25.0: # Use same tolerance as validation
                 cv.circle(vis_image, tuple(pos.astype(int)), 9, (0, 255, 255), -1) # Yellow for matched
            else:
                 cv.circle(vis_image, tuple(pos.astype(int)), 6, (0, 0, 255), -1) # Red for unmatched

    # --- Show the final image ---
    plt.figure(figsize=(12, 12))
    plt.imshow(cv.cvtColor(vis_image, cv.COLOR_BGR2RGB))
    plt.title(f"Validation Overlay for '{match_result['name'].upper()}' (Score: {match_result.get('final_score', 0.0):.2f})")
    plt.axis('off')
    plt.show()



# Add this function to helpers.py
import numpy as np
from scipy.spatial import cKDTree

def validate_and_score_match(detected_coords, match_result, distance_tolerance=25.0):
    """
    Performs a full geometric validation of a candidate match from the angle-based search.
    It checks how well the entire pattern fits the detected stars.
    """
    # --- 1. Get Initial Match Data and Perform Early Exit ---
    initial_angle_score = match_result['normalized_score']
    pattern = match_result['pattern']
    
    if initial_angle_score < 0.5 or match_result['anchor_idx'] == -1:
        match_result['final_score'] = 0.0
        return match_result

    scene_coords_array = np.array(detected_coords)
    scene_kdtree = cKDTree(scene_coords_array)

    # --- 2. Estimate the Full Transformation ... (No changes in this section)
    rotation = match_result['rotation_rad']
    scene_anchor_pos = scene_coords_array[match_result['anchor_idx']]
    pattern_anchor_node = match_result['pattern_anchor_node']
    pattern_anchor_pos = np.array(pattern_anchor_node.position)
    scale = 1.0
    if len(pattern_anchor_node.links) > 0:
        # ... (rest of scale estimation is unchanged)
        neighbor_label, link_data = next(iter(pattern_anchor_node.links.items()))
        pattern_dist_to_neighbor = link_data[2]
        expected_angle = _normalize_angle(link_data[3] + rotation)
        best_neighbor_idx, min_angle_diff = -1, float('inf')
        for i, coord in enumerate(detected_coords):
            if i == match_result['anchor_idx']: continue
            vec = coord - scene_anchor_pos
            angle_diff = abs(_normalize_angle(math.atan2(vec[1], vec[0]) - expected_angle))
            if angle_diff < min_angle_diff:
                min_angle_diff, best_neighbor_idx = angle_diff, i
        if best_neighbor_idx != -1 and pattern_dist_to_neighbor > 1e-6:
            scene_dist_to_neighbor = np.linalg.norm(scene_coords_array[best_neighbor_idx] - scene_anchor_pos)
            scale = scene_dist_to_neighbor / pattern_dist_to_neighbor

    # --- 3. Apply Transformation
    c, s = math.cos(rotation), math.sin(rotation)
    R = scale * np.array([[c, -s], [s, c]])
    t = scene_anchor_pos - (R @ pattern_anchor_pos)
    all_pattern_nodes_pos = np.array([node.position for node in pattern.nodes.values()])
    projected_nodes = (R @ all_pattern_nodes_pos.T).T + t
    match_result['transform'] = {'R': R, 't': t}

    MIN_AREA_THRESHOLD = 150 * 150 
    MAX_AREA_THRESHOLD = 5000 * 5000
    
    # Calculate the bounding box of the projected points
    min_coords = np.min(projected_nodes, axis=0)
    max_coords = np.max(projected_nodes, axis=0)
    
    width = max_coords[0] - min_coords[0]
    height = max_coords[1] - min_coords[1]
    area = width * height
    
    # If the area is too small, reject this match immediately by returning a zero score.
    if area < MIN_AREA_THRESHOLD or area > MAX_AREA_THRESHOLD:
        match_result['final_score'] = 0.0
        return match_result

    # --- 4. MODIFIED: Calculate Structural Fit Score with Harsher Penalty ---
    distances, _ = scene_kdtree.query(projected_nodes)
    num_matched_nodes = np.sum(distances < distance_tolerance)
    
    # Calculate the ratio of matched nodes
    match_ratio = num_matched_nodes / len(pattern.nodes)
    
    # Apply a non-linear penalty. Squaring the ratio punishes misses more heavily.
    # You can increase the exponent (e.g., to 3) for an even harsher penalty.
    structural_fit_score = match_ratio ** 2
    
    # --- 5. Calculate Final Score ---
    final_score = (0.4 * initial_angle_score) + (0.6 * structural_fit_score)
    match_result['final_score'] = final_score

    # --- 6. OPTIONAL: Refine Transform ... (No changes in this section)
    distances, scene_indices = scene_kdtree.query(projected_nodes)
    pattern_inliers = all_pattern_nodes_pos[distances < distance_tolerance]
    scene_inliers = scene_coords_array[scene_indices[distances < distance_tolerance]]
    if len(pattern_inliers) >= 3:
        refined_M, _ = cv.estimateAffinePartial2D(pattern_inliers, scene_inliers)
        if refined_M is not None:
            match_result['transform']['R'] = refined_M[:, :2]
            match_result['transform']['t'] = refined_M[:, 2]
    
    return match_result




def calculate_mask_fit_score(gray_sky_image, node_mask, line_mask, match_result):
    """
    Calculates a score based on how well the sky image pixels fit the pattern's masks,
    given a geometric transformation.
    """
    if 'transform' not in match_result:
        return 0.0

    # --- 1. Get the Inverse Transform ... (same as before) ...
    transform = match_result['transform']
    M = np.vstack([transform['R'], transform['t']]).T
    
    try:
        inv_M = cv.invertAffineTransform(M)
    except cv.error:
        return 0.0

    # --- 2. Warp the Sky Image ... (same as before) ...
    pattern_size = (node_mask.shape[1], node_mask.shape[0])
    warped_sky = cv.warpAffine(gray_sky_image, inv_M, pattern_size, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    # --- 3. Dilate Masks for Robustness ---
    # NEW: Create a small kernel and dilate the masks. This allows for slight
    # inaccuracies in the transformation.
    kernel = np.ones((5, 5), np.uint8) # 5x5 is a good starting point
    dilated_node_mask = cv.dilate(node_mask, kernel, iterations=1)
    dilated_line_mask = cv.dilate(line_mask, kernel, iterations=1)

    # --- 4. Calculate Scores Based on Dilated Masks ---
    node_mask_norm = dilated_node_mask / 255.0
    line_mask_norm = dilated_line_mask / 255.0
    
    num_node_pixels = np.sum(node_mask_norm)
    num_line_pixels = np.sum(line_mask_norm)

    if num_node_pixels < 1e-6:
        return 0.0

    node_score = np.sum(warped_sky * node_mask_norm) / num_node_pixels
    
    line_score = 0.0
    if num_line_pixels > 1e-6:
        # Subtract the node mask from the line mask to prevent stars that
        # are on a line from being penalized. We only want to measure the
        # brightness of the "empty" parts of the line.
        line_only_mask = np.clip(line_mask_norm - node_mask_norm, 0, 1)
        num_line_only_pixels = np.sum(line_only_mask)
        if num_line_only_pixels > 1e-6:
            line_score = np.sum(warped_sky * line_only_mask) / num_line_only_pixels

    mask_fit_score = (node_score - line_score) / 255.0
    
    # For debugging, you can add this block:
    # if match_result['name'] == 'mensa': # or some other constellation you are testing
    #     plt.figure(figsize=(10,5))
    #     plt.subplot(1,2,1); plt.imshow(warped_sky, cmap='gray'); plt.title("Warped Sky")
    #     plt.imshow(dilated_node_mask, cmap='Reds', alpha=0.5)
    #     plt.imshow(dilated_line_mask, cmap='Greens', alpha=0.5)
    #     plt.subplot(1,2,2); plt.imshow(gray_sky_image, cmap='gray'); plt.title("Original Sky")
    #     plt.show()

    return max(0, mask_fit_score)

def plot_match_in_scene(sky_image, all_coords, best_match):
    """
    Generates a plot showing all detected scene coordinates with the best-matched
    constellation pattern overlaid.

    Args:
        sky_image (np.array): The original sky image (can be BGR or grayscale).
        all_coords (list): The list of all (x,y) star coordinates detected in the scene.
        best_match (dict): The final result dictionary for the winning constellation.
    """
    # 1. Prepare a color image for plotting
    vis_image = cv.cvtColor(sky_image, cv.COLOR_GRAY2BGR) if len(sky_image.shape) == 2 else sky_image.copy()

    # 2. Plot all original coordinates from the scene
    # These are drawn as small, light blue circles for context.
    for x, y in all_coords:
        cv.circle(vis_image, (int(x), int(y)), 11, (0, 255, 0), 2)

    # 3. Check if a valid match and transform exist to be plotted
    # The 'transform' is calculated in the `validate_and_score_match` function.
    if best_match.get('final_score', 0.0) < 0.1 or 'transform' not in best_match:
        print("Match score is too low or transform is missing. Cannot plot overlay.")
        
        # If a basic anchor was found, we can at least highlight it
        if best_match.get('anchor_idx', -1) != -1:
            anchor_pos = all_coords[best_match['anchor_idx']]
            cv.circle(vis_image, (int(anchor_pos[0]), int(anchor_pos[1])), 12, (0, 0, 255), 2,
                      lineType=cv.LINE_AA)
            cv.putText(vis_image, "Anchor Found (Low Score)", (int(anchor_pos[0]) + 15, int(anchor_pos[1]) + 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # 4. A good match was found, so we plot the full pattern
        transform = best_match['transform']
        R, t = transform['R'], transform['t']
        pattern = best_match['pattern']
        
        # Project all of the pattern's nodes into the scene using the transform
        all_pattern_nodes_pos = np.array([node.position for node in pattern.nodes.values()])
        projected_nodes = (R @ all_pattern_nodes_pos.T).T + t
        
        # Create a map from a node's label (e.g., "N1") to its index for drawing edges
        label_to_idx = {label: i for i, label in enumerate(pattern.nodes.keys())}

        # 5. Draw the pattern's edges (lines)
        for start_label, end_label in pattern.edges:
            p1 = tuple(projected_nodes[label_to_idx[start_label]].astype(int))
            p2 = tuple(projected_nodes[label_to_idx[end_label]].astype(int))
            cv.line(vis_image, p1, p2, (0, 255, 0), 2, lineType=cv.LINE_AA) # Green lines
        
        # 6. Draw the pattern's nodes, color-coded by match quality
        scene_kdtree = cKDTree(np.array(all_coords))
        distances, _ = scene_kdtree.query(projected_nodes)
        
        for i, pos in enumerate(projected_nodes):
            # If a projected node is very close to a real detected star, it's a confirmed match
            if distances[i] < 25.0: # Use the same tolerance as in validation
                 # Draw matched nodes as bright yellow circles
                 cv.circle(vis_image, tuple(pos.astype(int)), 9, (0, 255, 255), -1, lineType=cv.LINE_AA)
            else:
                 # Draw unmatched nodes as smaller, translucent red circles
                 cv.circle(vis_image, tuple(pos.astype(int)), 6, (0, 0, 255), -1, lineType=cv.LINE_AA)

    # 7. Display the final plot using Matplotlib
    plt.figure(figsize=(14, 14))
    plt.imshow(cv.cvtColor(vis_image, cv.COLOR_BGR2RGB))
    title = (f"Match Result: '{best_match['name'].upper()}'\n"
             f"Overall Score: {best_match.get('overall_score', 0.0):.3f}")
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
