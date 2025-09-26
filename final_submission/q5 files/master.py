import cv2 as cv
import numpy as np
import helpers as h
import glob
import os
import patternsHelper as ph
from imageHelper import Image as imageHelper
import config

PATCH_FOLDER_NAME = "patches"
PATCHES_FILE_SUFFIX = "*.png"
SKY_FILE_SUFFIX = "*.tif"


def get_info_on_patterns(pattern_dir):
    patterns = []
    pattern_files = glob.glob(pattern_dir)
    for pattern_file in pattern_files:
        constellation_name = os.path.basename(pattern_file).replace('_pattern.png', '')
        if config.verbose:
            print(f"\tProcessing pattern: {constellation_name}")
        pattern_img = cv.imread(pattern_file, cv.IMREAD_UNCHANGED)
        pat, node_mask, line_mask = ph.extract_pattern_from_image(pattern_img, name=constellation_name)

        patterns.append((pat, node_mask, line_mask))

    return patterns


def find_constelation(sky_file, patch_dir, patterns):

    # Load sky image
    sky = imageHelper(sky_file, path_to_patches=patch_dir)

    sky.detect_stars_from_sky(thresh_value=180)

    sky.iterate_through_patches()
    coordinates = sky.get_coordinates()
    coords_black = sky.create_black_image_with_coordinates()

    if config.verbose:
        print(f"\nDetected {len(coordinates)} star coordinates from patches.")
    if len(coordinates) < 2:
        raise ValueError("Not enough stars detected to perform matching. Exiting.")
        

    #Turn pattern to binary mask
    angles = h.calculate_coordinate_angles(coordinates)
    if config.verbose:
        print("\nCalculating possible angles between star coordinates")
        print()

    #Print number of patterns found
    num_of_patches = len(coordinates)
    if config.verbose:
        print(f"\nNumber of star patches detected: {num_of_patches}")
    
    all_validated_matches = []
    if config.verbose:
        print("\n[Stage 1 & 2] Generating & Validating Top Candidates for Each Pattern:")

    for pat, node_mask, line_mask in patterns:
        anchor_nodes = pat.get_highest_degree_nodes()
        if not anchor_nodes:
            continue

        size = pat.get_number_of_nodes()
        if config.verbose:
            print(f"\nPattern '{pat.name}' with {size} nodes and {len(anchor_nodes)} anchor nodes (highest degree).")


        if size > num_of_patches:
            if config.verbose:
                print(f"  - Skipping pattern '{pat.name}' as it has more nodes ({size}) than detected stars ({num_of_patches}).")
            continue

        # Generate all possible initial matches from all anchors
        all_initial_candidates = []
        for anchor_node in anchor_nodes:
            # The new function returns a list of candidates
            initial_matches = h.find_best_match_by_angles(pat, coordinates, anchor_node, angle_tolerance_deg=5.0)
            all_initial_candidates.extend(initial_matches)
        
        if not all_initial_candidates:
            if config.verbose:
                print(f"  - No initial candidates found for '{pat.name}'.")
            continue
            
        # Greedy approach does not work well here, so we will sort in order of normalized_score just as an initial guess and validate all
        all_initial_candidates.sort(key=lambda x: x['normalized_score'], reverse=True)

        # Validate each of these top candidates
        best_validated_match_for_pattern = None
        for candidate in all_initial_candidates:
            validated_match = h.validate_and_score_match(coordinates, candidate, distance_tolerance=20.0)
            
            if best_validated_match_for_pattern is None or validated_match['final_score'] > best_validated_match_for_pattern['final_score']:
                best_validated_match_for_pattern = validated_match
        
        # Add mask info and append the single best result for this pattern
        best_validated_match_for_pattern['node_mask'] = node_mask
        best_validated_match_for_pattern['line_mask'] = line_mask
        all_validated_matches.append(best_validated_match_for_pattern)
        
        if config.verbose:
            print(f"  - Best validated match for '{pat.name}': Final Score = {best_validated_match_for_pattern['final_score']:.2f}")
            
    if config.verbose:
        print("\n[Stage 3] Refining scores with pixel-level mask fitting:")
    final_results = []
    for match in all_validated_matches:
        # Generate maks_score using template matching
        score = cv.matchTemplate(coords_black, match['line_mask'], cv.TM_CCOEFF_NORMED)
        _, mask_score, _, loc = cv.minMaxLoc(score)   
        match['mask_fit_score'] = mask_score

        # Calculate sparsity score
        sparsity_score = h.calculate_sparsity_score(coordinates, match)
        match['sparsity_score'] = sparsity_score
        
        geometric_score = match.get('final_score', 0.0)
        match['overall_score'] = geometric_score
        final_results.append(match)

        if config.verbose:
            print(f"  - Constellation: {match['name']:<15} | Geo Score: {geometric_score:.2f} | Mask Score: {mask_score:.2f} | Sparsity Score: {sparsity_score:.2f} | Overall: {match['overall_score']:.2f}")

    # Use Geometric score to determine best match
    # Add mask_fit_score as tie breaker if multiple have same highest overall_score
    
    best_match = max(final_results, key=lambda r: (r.get('overall_score', 0.0), r.get('sparsity_score', 0.0)))
    
    if config.verbose:
        print("\n" + "="*25)
        print("      FINAL RESULT")
        print("="*25)
        
        print("\nProjected coordinates of the matched constellation:")
    
    if 'transform' in best_match and best_match['transform'] is not None:
        transform = best_match['transform']
        R, t = transform['R'], transform['t']
        pattern = best_match['pattern']
        pattern_name = best_match['name']
        confidence = best_match['overall_score']

        sky.set_constelation(pattern_name)
        sky.set_confidence_score(confidence)

        if config.verbose:
            print(f"Constellation: {pattern_name}")
            print(f"Overall Confidence Score: {confidence:.2f}")
        
        # Get all original pattern nodes, sorted by label for consistency
        all_pattern_nodes = sorted(pattern.nodes.values(), key=lambda n: n.label)
        all_pattern_nodes_pos = np.array([node.position for node in all_pattern_nodes])
        
        # Apply the final transform to get the coordinates in the sky image
        projected_nodes = (R @ all_pattern_nodes_pos.T).T + t
        
        for i, proj_coord in enumerate(projected_nodes):
            original_node = all_pattern_nodes[i]
            if config.verbose:
                print(f"  - Node {original_node.label}: (x={proj_coord[0]:.1f}, y={proj_coord[1]:.1f})")
    else:
        if config.verbose:
            print("  - No valid transform was found to project coordinates.")


    patch_dict = sky.get_patch_dict()
    if config.verbose:
        print("\nPatch Matching Results:")
        for patch_name, coord in patch_dict.items():
            print(f"  - {patch_name}: Coordinates = {coord}")

    constelation = sky.get_constelation()
    confidence_score = sky.get_confidence_score()

    if config.verbose:
        print(f"\nDetected Constellation: {constelation} with Confidence Score: {confidence_score:.2f}")


    # We use sky.image, which is the original BGR image loaded at the start
    if config.verbose:
        h.plot_match_in_scene(sky.image, coordinates, best_match)
    
    return patch_dict, constelation, confidence_score


def master_function(folder, patterns):
    if config.verbose:
        print(f"\nProcessing folder: {folder}")
    
    sky_image = glob.glob(os.path.join(folder, SKY_FILE_SUFFIX))[0]
    if config.verbose:
        print(f"Sky image file: {sky_image}")

    patch_dir = os.path.join(folder, PATCH_FOLDER_NAME, PATCHES_FILE_SUFFIX)
    if config.verbose:
        print(f"Patch directory: {patch_dir}")

    patch_dict, constelation, confidence_score = find_constelation(sky_image, patch_dir, patterns)
    cv.destroyAllWindows()

    return patch_dict, constelation, confidence_score



    