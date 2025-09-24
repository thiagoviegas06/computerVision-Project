import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import glob
import os
import patternsHelper as ph
from imageHelper import Image as imageHelper

def main(sky_file, pattern_dir, patch_dir):

    # Load sky image
    mensa = imageHelper(sky_file, path_to_patches=patch_dir)
    stars_with_sizes = mensa.detect_stars_from_sky(thresh_value=180)

    mensa.iterate_through_patches()

    coordinates = mensa.get_coordinates()

    sky_gray = cv.cvtColor(mensa.image, cv.COLOR_BGR2GRAY)
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
        pat, node_mask, line_mask = ph.extract_pattern_from_image(pattern_img, name=constellation_name)

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
            
        # 2. Prune by keeping only the top N unique candidates based on score
        # This prevents us from validating hundreds of similar, poor-quality matches
        all_initial_candidates.sort(key=lambda x: x['normalized_score'], reverse=True)
        top_candidates_to_validate = all_initial_candidates[:5] # Validate the top 5 hypotheses

        # 3. Validate each of these top candidates
        best_validated_match_for_pattern = None
        for candidate in top_candidates_to_validate:
            validated_match = h.validate_and_score_match(coordinates, candidate, distance_tolerance=25.0)
            
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
        mask_score = h.calculate_mask_fit_score(sky_gray, match['node_mask'], match['line_mask'], match)
        match['mask_fit_score'] = mask_score
        
        geometric_score = match.get('final_score', 0.0)
        match['overall_score'] = (0.7 * geometric_score) + (0.3 * mask_score)
        final_results.append(match)

        print(f"  - Constellation: {match['name']:<15} | Geo Score: {geometric_score:.2f} | Mask Score: {mask_score:.2f} | Overall: {match['overall_score']:.2f}")

    # --- Determine the Best Match based on the OVERALL Score ---
    best_match = max(final_results, key=lambda r: r.get('overall_score', 0.0))
    
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
    h.plot_match_in_scene(mensa.image, coordinates, best_match)
    
    # pattern = cv.imread('patterns/mensa_pattern.png', cv.IMREAD_UNCHANGED)
    # pat, node_mask, line_mask = ph.extract_pattern_from_image(pattern, name="Mensa")

    # print("Nodes:", len(pat.nodes))
    # for lbl, node in pat.nodes.items():
    #     print(lbl, node.position, "deg:", len(node.links))

    # print("Edges:", pat.edges)  # list of (a,b)

    # highest_deg_node = pat.getHighestDegreeNode()

    # print("Anchor node:", highest_deg_node.label, highest_deg_node.position, "deg:", len(highest_deg_node.links))

   


if __name__ == "__main__":
    sky_images = [ 'train/Pisces/pisces_image.tif',
                   'train/Mensa/mensa_image.tif',
                   'train/Taurus/taurus_image.tif' ]
    patch_dirs = [ 'train/Pisces/patches/*.png',
                    'train/Mensa/patches/*.png',
                    'train/Taurus/patches/*.png' ]
    pattern_dir = 'patterns/*.png'
    for sky_file, patch_dir in zip(sky_images, patch_dirs):
        print(f"\n\nProcessing sky image: {sky_file}")
        main(sky_file, pattern_dir, patch_dir)
        # Delete all OpenCV windows to avoid overlap
        cv.destroyAllWindows()
        
    # main('train/Mensa/mensa_image.tif', 'patterns/*.png', 'train/Mensa/patches/*.png')
    