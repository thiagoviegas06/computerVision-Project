import cv2 as cv
import math, numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from patternsHelper import Node, Pattern
import config
from scipy.spatial import cKDTree
import re
from pathlib import Path
from typing import List
import os

#  Prompted AI for help with sorting paths function
def sort_by_number_first(folder_list: List[Path]) -> List[Path]:
    """
    Sorts a list of Path objects primarily by the first number found in the
    name, and secondarily by the alphabetical name (case-insensitive).

    Folders without any numbers are placed at the end of the list.

    Args:
        folder_list: A list of pathlib.Path objects.

    Returns:
        A new list containing the sorted pathlib.Path objects.
    """
    def get_sort_key(path: Path):
        name = path.name
        # Find the first sequence of digits in the folder name
        match = re.search(r'(\d+)', name)

        if match:
            # If a number is found, return a tuple: (number, name)
            # The number is the primary sort key.
            # The name is the secondary (tie-breaker) sort key.
            return (int(match.group(1)), name.lower())
        else:
            # If no number is found, use 'infinity' as the primary key.
            # This pushes these items to the end of the sorted list.
            # The name is still the secondary key to sort them among themselves.
            return (float('inf'), name.lower())

    return sorted(folder_list, key=get_sort_key)

def equalize_image(image):
    # First, converting image to yCbCr
    image_yCbCr = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    # Apply Histogram Equalization
    image_yCbCr[:,:,0] = cv.equalizeHist(image_yCbCr[:,:,0])

    # Convert back to BGR
    equalized_image = cv.cvtColor(image_yCbCr, cv.COLOR_YCrCb2RGB)

    return equalized_image

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

def draw_lines_between_all_points(image, pointList, color=(255, 255, 255), thickness=2):
    # Ensure it's an ndarray
    pointList = np.array(pointList)

    if len(pointList) == 0:   
        return image

    for i in range(len(pointList) - 1):
        for j in range(i + 1, len(pointList)):
            if config.verbose:
                print(f"Drawing line between point {i} and point {j}")
            cv.line(image,
                    tuple(pointList[i].astype(int)),
                    tuple(pointList[j].astype(int)),
                    color, thickness)

    return image

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

    # AI suggested we use a  k-d tree for super-fast nearest neighbor searches
    #   A k-d tree is a binary tree that recursively partitions space into halves. 
    #   Points are stored either to the right or left of a splitting plane. 
    #   This will allow us to quickly find the nearest neighbors of any point.
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
    # For each top candidate, look for a 1-to-1 match to get an accurate score
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
            
            # If a match is found within tolerance, count it and add the detected angle
            if best_match_idx != -1 and min_diff <= tolerance_rad:
                current_matches += 1
                temp_detected_sig.pop(best_match_idx)

        # The score is now a true match count, which cannot exceed len(pattern_sig)
        candidates.append({'rotation_rad': avg_rotation, 'score': current_matches})
        
    return candidates


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
        if config.verbose:
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

def validate_and_score_match(detected_coords, match_result, distance_tolerance=25.0):
    """
    Performs a full geometric validation of a candidate match from the angle-based search.
    It checks how well the entire pattern fits the detected stars.
    """
    # Get Initial Match Data and Perform Early Exit
    initial_angle_score = match_result['normalized_score']
    pattern = match_result['pattern']
    
    if initial_angle_score < 0.5 or match_result['anchor_idx'] == -1:
        match_result['final_score'] = 0.0
        return match_result

    scene_coords_array = np.array(detected_coords)
    scene_kdtree = cKDTree(scene_coords_array)

    #  Estimate the Full Transformation
    rotation = match_result['rotation_rad']
    scene_anchor_pos = scene_coords_array[match_result['anchor_idx']]
    pattern_anchor_node = match_result['pattern_anchor_node']
    pattern_anchor_pos = np.array(pattern_anchor_node.position)
    scale = 1.0
    if len(pattern_anchor_node.links) > 0:
        # Use the first linked neighbor to estimate scale
        neighbor_label, link_data = next(iter(pattern_anchor_node.links.items()))
        pattern_dist_to_neighbor = link_data[2]
        expected_angle = _normalize_angle(link_data[3] + rotation)
        best_neighbor_idx, min_angle_diff = -1, float('inf')
        # Find the best matching neighbor in the scene
        for i, coord in enumerate(detected_coords):
            if i == match_result['anchor_idx']: continue
            vec = coord - scene_anchor_pos
            angle_diff = abs(_normalize_angle(math.atan2(vec[1], vec[0]) - expected_angle))
            if angle_diff < min_angle_diff:
                min_angle_diff, best_neighbor_idx = angle_diff, i
        if best_neighbor_idx != -1 and pattern_dist_to_neighbor > 1e-6:
            scene_dist_to_neighbor = np.linalg.norm(scene_coords_array[best_neighbor_idx] - scene_anchor_pos)
            scale = scene_dist_to_neighbor / pattern_dist_to_neighbor

    # Apply Transformation and Project All Pattern Nodes
    c, s = math.cos(rotation), math.sin(rotation)

    # R = rotation matrix, t = translation vector
    R = scale * np.array([[c, -s], [s, c]])
    t = scene_anchor_pos - (R @ pattern_anchor_pos)

    # All pattern nodes projected into the scene
    all_pattern_nodes_pos = np.array([node.position for node in pattern.nodes.values()])
    projected_nodes = (R @ all_pattern_nodes_pos.T).T + t

    # Store the transform for potential later use
    match_result['transform'] = {'R': R, 't': t}

    MIN_AREA_THRESHOLD = 150 * 150 
    MAX_AREA_THRESHOLD = 5500 * 5500
    
    # Calculate the bounding box of the projected points
    min_coords = np.min(projected_nodes, axis=0)
    max_coords = np.max(projected_nodes, axis=0)
    
    width = max_coords[0] - min_coords[0]
    height = max_coords[1] - min_coords[1]
    area = width * height
    
    # If the area is too small or too big, reject this match immediately by returning a zero score.
    if area < MIN_AREA_THRESHOLD or area > MAX_AREA_THRESHOLD:
        match_result['final_score'] = 0.0
        return match_result

    # Calculate Structural Fit Score with Harsher Penalty
    distances, _ = scene_kdtree.query(projected_nodes)
    num_matched_nodes = np.sum(distances < distance_tolerance)
    
    # Calculate the ratio of matched nodes
    match_ratio = num_matched_nodes / len(pattern.nodes)
    match_result['num_matched_nodes'] = num_matched_nodes

    # Apply a non-linear penalty. Squaring the ratio punishes misses more heavily.
    # You can increase the exponent (e.g., to 3) for an even harsher penalty.
    match_ratio = num_matched_nodes / len(pattern.nodes)
    structural_fit_score = match_ratio ** 2
    
    # Calculate Final Score
    final_score = (0.4 * initial_angle_score) + (0.6 * structural_fit_score)
    match_result['final_score'] = final_score

    # Prompting AI to help us with the overall accuracy of the function it told us to 
    # refine transform Using All Inliers
    #   This step can help correct small errors in rotation/scale/translation.
    #   We use only the inliers (those within distance_tolerance) for refinement.
    #   This is a simple least-squares fit using OpenCV's estimateAffinePartial2D.
    #   It returns a 2x3 affine matrix, from which we extract R and t
    #  Note: This step is optional and can be skipped for speed if needed as it helps with futre uses of R and t.
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

    # Get the Inverse Transform Matrix
    transform = match_result['transform']
    M = np.vstack([transform['R'], transform['t']]).T
    
    try:
        inv_M = cv.invertAffineTransform(M)
    except cv.error:
        return 0.0

    # Warp the Sky Image to Align with the Pattern
    pattern_size = (node_mask.shape[1], node_mask.shape[0])
    warped_sky = cv.warpAffine(gray_sky_image, inv_M, pattern_size, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    # Dilate Masks for Robustness
    # Create a small kernel and dilate the masks. This allows for slight
    # inaccuracies in the transformation.
    kernel = np.ones((5, 5), np.uint8) # 5x5 is a good starting point
    dilated_node_mask = cv.dilate(node_mask, kernel, iterations=1)
    dilated_line_mask = cv.dilate(line_mask, kernel, iterations=1)

    # Calculate Scores Based on Dilated Masks
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
    # Prepare a color image for plotting
    vis_image = cv.cvtColor(sky_image, cv.COLOR_GRAY2BGR) if len(sky_image.shape) == 2 else sky_image.copy()

    # Plot all original coordinates from the scene
    # These are drawn as small, light blue circles for context.
    for x, y in all_coords:
        cv.circle(vis_image, (int(x), int(y)), 11, (0, 255, 0), 2)

    # Check if a valid match and transform exist to be plotted
    # The 'transform' is calculated in the `validate_and_score_match` function.
    if best_match.get('final_score', 0.0) < 0.1 or 'transform' not in best_match and config.verbose:
        print("Match score is too low or transform is missing. Cannot plot overlay.")
        
        # If a basic anchor was found, we can at least highlight it
        if best_match.get('anchor_idx', -1) != -1:
            anchor_pos = all_coords[best_match['anchor_idx']]
            cv.circle(vis_image, (int(anchor_pos[0]), int(anchor_pos[1])), 12, (0, 0, 255), 2,
                      lineType=cv.LINE_AA)
            cv.putText(vis_image, "Anchor Found (Low Score)", (int(anchor_pos[0]) + 15, int(anchor_pos[1]) + 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # A good match was found, so we plot the full pattern
        transform = best_match['transform']
        R, t = transform['R'], transform['t']
        pattern = best_match['pattern']
        
        # Project all of the pattern's nodes into the scene using the transform
        all_pattern_nodes_pos = np.array([node.position for node in pattern.nodes.values()])
        projected_nodes = (R @ all_pattern_nodes_pos.T).T + t
        
        # Create a map from a node's label (e.g., "N1") to its index for drawing edges
        label_to_idx = {label: i for i, label in enumerate(pattern.nodes.keys())}

        # Draw the pattern's edges (lines)
        for start_label, end_label in pattern.edges:
            p1 = tuple(projected_nodes[label_to_idx[start_label]].astype(int))
            p2 = tuple(projected_nodes[label_to_idx[end_label]].astype(int))
            cv.line(vis_image, p1, p2, (0, 255, 0), 2, lineType=cv.LINE_AA) # Green lines
        
        # Draw the pattern's nodes, color-coded by match quality
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

    # Display the final plot using Matplotlib
    plt.figure(figsize=(14, 14))
    plt.imshow(cv.cvtColor(vis_image, cv.COLOR_BGR2RGB))
    title = (f"Match Result: '{best_match['name'].upper()}'\n"
             f"Overall Score: {best_match.get('overall_score', 0.0):.3f}")
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()


def calculate_sparsity_score(all_detected_coords, match_result, distance_tolerance=25.0):
    """
    Calculates a precision score based on how many of the stars within a match's
    bounding box are successfully accounted for by the pattern. This version
    ensures the score cannot exceed 1.0.
    """
    if 'transform' not in match_result or match_result['transform'] is None:
        return 0.0

    transform = match_result['transform']
    R, t = transform['R'], transform['t']
    pattern = match_result['pattern']
    
    # Define the local area by finding the pattern's bounding box on the image
    all_pattern_nodes_pos = np.array([node.position for node in pattern.nodes.values()])
    projected_nodes = (R @ all_pattern_nodes_pos.T).T + t
    min_coords = np.min(projected_nodes, axis=0)
    max_coords = np.max(projected_nodes, axis=0)

    # Find all detected stars that fall strictly within this bounding box
    stars_in_bbox = [
        coord for coord in all_detected_coords 
        if (min_coords[0] <= coord[0] <= max_coords[0]) and \
           (min_coords[1] <= coord[1] <= max_coords[1])
    ]
    
    num_stars_in_bbox = len(stars_in_bbox)
    if num_stars_in_bbox == 0:
        return 0.0

    # Count how many pattern nodes match a star *from this local subset*
    # This is the crucial step that guarantees consistency.
    stars_in_bbox_array = np.array(stars_in_bbox)
    kdtree_in_bbox = cKDTree(stars_in_bbox_array)
    
    distances, _ = kdtree_in_bbox.query(projected_nodes)
    num_matched_stars_in_bbox = np.sum(distances < distance_tolerance)

    # The score is the ratio of matched stars to total stars *in the same defined area*.
    score = num_matched_stars_in_bbox / num_stars_in_bbox
    
    return score