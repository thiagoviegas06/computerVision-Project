"""
This modeule contains helper functions and classes for constellation pattern extraction and matching.
Authors: Diego Rosenberg (dr3432), Thiago Viegas (tjv235)

"""

"""Importing necessary libraries."""
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Tuple, List
import glob
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
import math
import re
from scipy.spatial import cKDTree

# Argument Parser Function
def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    
    # Positional argument: root folder
    parser.add_argument(
        "root_folder",
        type=str,
        help="Path to the dataset root (contains patterns/, train/, validation/, etc.)."
    )

    # -f <folder name> argument (required)
    parser.add_argument(
        "-f", "--folder",
        type=str,
        required=True,
        help="Target folder to process (such as validation or test)."
    )

    # Optional verbose flag
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Add print statements and plotting for debugging purposes."
    )

    return parser.parse_args()


# Type alias for 2D vector
Vec2 = Tuple[float, float]

"""Definition of Data Clasesses for constellation pattern extraction and matching."""
# Create a Node class to represent each star in the pattern
@dataclass
class Node:
    label: str
    position: Vec2
    size: float = 0.0  
    # neighbor -> (unit_dx, unit_dy, length, angle)
    links: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)

    def add_link(self, other: "Node"):
        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]
        L  = math.hypot(dx, dy)
        if L < 1e-9: return
        ux, uy = dx/L, dy/L
        ang = math.atan2(dy, dx)
        self.links[other.label] = (ux, uy, L, ang)

# Create a Pattern class to hold the constellation pattern
@dataclass
class Pattern:
    name: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # undirected (a,b)

    def add_node(self, label: str, pos: Vec2, size: float = 0.0): # <-- UPDATE THIS METHOD
        self.nodes[label] = Node(label, pos, size=size)

    def add_edge(self, a: str, b: str):
        if a == b or (a, b) in self.edges or (b, a) in self.edges: return
        self.edges.append((a, b))
        self.nodes[a].add_link(self.nodes[b])
        self.nodes[b].add_link(self.nodes[a])

    def getHighestDegreeNode(self) -> Node:
        best = None
        best_deg = -1
        for node in self.nodes.values():
            deg = len(node.links)
            if deg > best_deg:
                best = node
                best_deg = deg
        return best
    
    def get_highest_degree_nodes(self) -> List[Node]:
        """
        Finds all nodes with the maximum number of links (degree).
        Returns a list of nodes.
        """
        if not self.nodes:
            return []

        best_deg = -1
        # First pass: find the maximum degree in the pattern
        for node in self.nodes.values():
            best_deg = max(best_deg, len(node.links))
        
        if best_deg == -1:
            return []

        # Second pass: collect all nodes that have this maximum degree
        highest_degree_nodes = [
            node for node in self.nodes.values() if len(node.links) == best_deg
        ]
        return highest_degree_nodes


"""General Helper Functions for Code"""
def extract_pattern_from_image(bgr_or_rgba, name="pattern",
                               min_node_area=5, max_node_area=5_000,
                               green_hsv=((35, 40, 40), (90, 255, 255)),
                               white_hsv=((0, 0, 200), (180, 50, 255)),
                               link_hit_thresh=0.75,   # ≥ 75% of samples must be green
                               max_link_gap=5,         # max consecutive non-green samples
                               max_pair_px=99999       # (optionally limit how far nodes can connect)
                               ) -> Pattern:
    img = bgr_or_rgba
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    elif img.shape[2] == 3:
        bgr = img
        alpha = None
    else:
        raise ValueError("Input image must have 3 (BGR) or 4 (BGRA) channels")

    # HSV segmentation
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    white = cv.inRange(hsv, white_hsv[0], white_hsv[1])
    green = cv.inRange(hsv, green_hsv[0], green_hsv[1])

    # Clean masks a bit
    white = cv.morphologyEx(white, cv.MORPH_OPEN,  np.ones((3,3), np.uint8))
    white = cv.morphologyEx(white, cv.MORPH_DILATE, np.ones((3,3), np.uint8))
    green = cv.morphologyEx(green, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # Nodes from connected components
    num, labels, stats, cents = cv.connectedComponentsWithStats(white, connectivity=8)
    node_data = []
    for i in range(1, num):
        area = stats[i, cv.CC_STAT_AREA]
        if min_node_area <= area <= max_node_area:
            cx, cy = cents[i]  # float (x,y)
            node_data.append({'pos': (float(cx), float(cy)), 'area': float(area)})

    node_data.sort(key=lambda d: (d['pos'][1], d['pos'][0]))

    # Build pattern with nodes
    pat = Pattern(name=name)
    for k, data in enumerate(node_data, start=1):
        pat.add_node(f"N{k}", data['pos'], size=data['area'])

    # Helper: sample along a segment to check if it's painted green
    def link_exists(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L > max_pair_px: 
            return False
        n = max(2, int(L))  # 1px step
        xs = np.linspace(x1, x2, n).astype(np.int32)
        ys = np.linspace(y1, y2, n).astype(np.int32)
        xs = np.clip(xs, 0, green.shape[1]-1)
        ys = np.clip(ys, 0, green.shape[0]-1)
        line_vals = green[ys, xs] > 0
        hit_rate = line_vals.mean()
        # also ensure there isn't a long break
        gaps = 0
        max_gap_seen = 0
        for v in line_vals:
            if v: gaps = 0
            else:
                gaps += 1
                max_gap_seen = max(max_gap_seen, gaps)
        return (hit_rate >= link_hit_thresh) and (max_gap_seen <= max_link_gap)

    # Create edges by probing pairs (O(N^2) is fine for small constellations)
    labels = list(pat.nodes.keys())
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a, b = labels[i], labels[j]
            p1 = pat.nodes[a].position
            p2 = pat.nodes[b].position
            if link_exists(p1, p2):
                pat.add_edge(a, b)


    return pat, white, green


""" Definition of Classes for constellation pattern extraction and matching."""
# Image class for handling sky images and star detection
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

            if score > 0.95:
               self.coordinates_list.append(coordinates[0])
    
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


def categorize():
    return

"""def main(sky_file, pattern_dir, patch_dir):


    # Load sky image
    sky = imageHelper(sky_file, path_to_patches=patch_dir)

    sky.iterate_through_patches()
    coordinates = sky.get_coordinates()
    coords_black = sky.create_black_image_with_coordinates()

    sky_gray = cv.cvtColor(sky.image, cv.COLOR_BGR2GRAY)
    print(f"\nDetected {len(coordinates)} star coordinates from patches.")
    if len(coordinates) < 2:
        print("Not enough stars detected to perform matching. Exiting.")
        return

    #Turn pattern to binary mask
    angles = h.calculate_coordinate_angles(coordinates)
    print(angles)

    pattern_files = glob.glob(pattern_dir)
    patterns = []
    for pattern_file in pattern_files:
        constellation_name = os.path.basename(pattern_file).replace('_pattern.png', '')
        print(f"\nProcessing pattern: {constellation_name}")
        pattern_img = cv.imread(pattern_file, cv.IMREAD_UNCHANGED)
        pat, node_mask, line_mask = extract_pattern_from_image(pattern_img, name=constellation_name)

        patterns.append((pat, node_mask, line_mask))


    all_validated_matches = []
    print("\n[Stage 1 & 2] Generating & Validating Top Candidates for Each Pattern...")

    for pat, node_mask, line_mask in patterns:
        anchor_nodes = pat.get_highest_degree_nodes()
        if not anchor_nodes:
            continue

        # 1. Generate all possible initial matches from all anchors
        all_initial_candidates = []
        for anchor_node in anchor_nodes:
            # The new function returns a list of candidates
            initial_matches = h.find_best_match_by_angles(pat, coordinates, anchor_node, angle_tolerance_deg=5.0)
            all_initial_candidates.extend(initial_matches)
        
        if not all_initial_candidates:
            print(f"  - No initial candidates found for '{pat.name}'.")
            continue
            
        # Greedy approach does not work well here, so we will sort in order of normalized_score just as an initial guess and validate all
        all_initial_candidates.sort(key=lambda x: x['normalized_score'], reverse=True)

        # 3. Validate each of these top candidates
        best_validated_match_for_pattern = None
        for candidate in all_initial_candidates:
            validated_match = h.validate_and_score_match(coordinates, candidate, distance_tolerance=20.0)
            
            if best_validated_match_for_pattern is None or validated_match['final_score'] > best_validated_match_for_pattern['final_score']:
                best_validated_match_for_pattern = validated_match
        
        # Add mask info and append the single best result for this pattern
        best_validated_match_for_pattern['node_mask'] = node_mask
        best_validated_match_for_pattern['line_mask'] = line_mask
        all_validated_matches.append(best_validated_match_for_pattern)
        
        print(f"  - Best validated match for '{pat.name}': Final Score = {best_validated_match_for_pattern['final_score']:.2f}")


    print("\n[Stage 3] Refining scores with pixel-level mask fitting...")
    final_results = []
    for match in all_validated_matches:
        score = cv.matchTemplate(coords_black, match['line_mask'], cv.TM_CCOEFF_NORMED)
        _, mask_score, _, loc = cv.minMaxLoc(score)   

        # mask_score = h.calculate_mask_fit_score(sky_gray, match['node_mask'], match['line_mask'], match)
        match['mask_fit_score'] = mask_score

        sparsity_score = h.calculate_sparsity_score(coordinates, match)
        match['sparsity_score'] = sparsity_score
        
        geometric_score = match.get('final_score', 0.0)
        match['overall_score'] = (1 * geometric_score) + (0 * mask_score)
        final_results.append(match)

        print(f"  - Constellation: {match['name']:<15} | Geo Score: {geometric_score:.2f} | Mask Score: {mask_score:.2f} | Sparsity Score: {sparsity_score:.2f} | Overall: {match['overall_score']:.2f}")

    # --- Determine the Best Match based on the OVERALL Score ---
    # Add mask_fit_score as tie breaker if multiple have same overall_score
    
    best_match = max(final_results, key=lambda r: (r.get('overall_score', 0.0), r.get('sparsity_score', 0.0)))
    
    print("\n" + "="*25)
    print("      FINAL RESULT")
    print("="*25)
    
    print("\nProjected coordinates of the matched constellation:")
    if 'transform' in best_match and best_match['transform'] is not None:
        transform = best_match['transform']
        R, t = transform['R'], transform['t']
        pattern = best_match['pattern']
        
        # Get all original pattern nodes, sorted by label for consistency
        all_pattern_nodes = sorted(pattern.nodes.values(), key=lambda n: n.label)
        all_pattern_nodes_pos = np.array([node.position for node in all_pattern_nodes])
        
        # Apply the final transform to get the coordinates in the sky image
        projected_nodes = (R @ all_pattern_nodes_pos.T).T + t
        
        for i, proj_coord in enumerate(projected_nodes):
            original_node = all_pattern_nodes[i]
            print(f"  - Node {original_node.label}: (x={proj_coord[0]:.1f}, y={proj_coord[1]:.1f})")
    else:
        print("  - No valid transform was found to project coordinates.")

    # We use sky.image, which is the original BGR image loaded at the start
    h.plot_match_in_scene(sky.image, coordinates, best_match)

    return"""




def main(root_folder: str, folder_name: str, verbose: bool):
    if verbose:
        import matplotlib.pyplot as plt

    # Handle all patterns first (we only need to load them once)
    pattern_dir = os.path.join(root_folder, "patterns", "*.png")
    pattern_files = glob.glob(pattern_dir)
    patterns = []
    for pattern_file in pattern_files:
        constellation_name = os.path.basename(pattern_file).replace('_pattern.png', '')
        if verbose:
            print(f"Processing pattern: {constellation_name}")
        pattern_img = cv.imread(pattern_file, cv.IMREAD_UNCHANGED)
        pat, node_mask, line_mask = extract_pattern_from_image(pattern_img, name=constellation_name)

        patterns.append((pat, node_mask, line_mask))

    # Handle all folders within folder_name
    target_dir = os.path.join(root_folder, folder_name)

    ####### Get all subdirs to process (PLEASE TRY TO SORT THEM IN A WAY THAT WORKS, ASK THE TAs if this is necesssary as well)  
    subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    subdirs = sorted(subdirs)
    print(f"Found {len(subdirs)} subdirectories in '{target_dir}': {subdirs}")
    for subdir in subdirs:
        print(f"Processing subdirectory: {subdir}")
        sky_file = os.path.join(target_dir, subdir, f"{subdir.lower()}_image.tif")
        patch_dir = os.path.join(target_dir, subdir, "patches", "*.png")
        if not os.path.exists(sky_file):
            print(f"Sky image not found at {sky_file}. Skipping this subdirectory.")
            continue
        if not glob.glob(patch_dir):
            print(f"No patches found in {patch_dir}. Skipping this subdirectory.")
            continue

        print(sky_file, pattern_dir, patch_dir)
        print()





if __name__ == "__main__":
    args = parse_args()
    # Example usage of parsed arguments
    root_folder = args.root_folder
    folder_name = args.folder
    verbose = args.verbose
    if verbose:
        print(f"Root folder: {root_folder}")
        print(f"Target folder: {folder_name}")
        print(f"Verbose mode enabled.")
    
    main(root_folder, folder_name, verbose)