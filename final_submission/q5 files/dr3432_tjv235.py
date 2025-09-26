#!/usr/bin/env python3
"""
Student Template for Constellation Classification
Replace 'netid' with your actual NetID (e.g., abc123.py)

REQUIREMENTS:
1. Command line: python netid.py <root_folder> -f <folder_name> [-v]
2. Output CSV: {folder_name}_results.csv saved in same directory as this script
3. CSV format: See specification below
"""

import pandas as pd
import argparse
import sys
import os
from pathlib import Path
import master
import config
import helpers as h

PATTERN_FOLDER_NAME = "patterns"
PATTERN_FILE_SUFFIX = "*.png"  # Pattern files in patterns/ folder
TEAM_FOLDER_NAME = "dr3432_tjv235"  

def process_constellation_data(root_folder, target_folder):
    """
    Main processing function - implement your constellation classification here
    
    Args:
        root_folder: Path to root data directory (contains patterns/, test/, etc.)
        target_folder: Folder to process ('test', 'validation', etc.)
        config.verbose: Whether to print detailed output
    
    Returns:
        pandas.DataFrame: Results in required CSV format
    """
    
    root_path = root_folder
    if not os.path.exists(root_folder):
        raise ValueError(f"Root folder '{root_folder}' does not exist.")
    root_folder = Path(root_folder)
    
    target_dir = os.path.join(root_path, target_folder)
    if not os.path.exists(target_dir):
        raise ValueError(f"Target folder '{target_dir}' does not exist.")
    target_folder = root_folder / target_dir

    patterns_path = os.path.join(root_path, PATTERN_FOLDER_NAME)
    if not os.path.exists(patterns_path):
        raise ValueError(f"Patterns folder '{patterns_path}' does not exist.")
    
    patterns_folder = root_folder / PATTERN_FOLDER_NAME

    if config.verbose:
        print(f"Root folder: {root_path}")
        print(f"Target folder: {target_dir}")
        print(f"Patterns folder: {patterns_path}")
    
    if config.verbose:
        print("Loading patterns...")
    patterns = master.get_info_on_patterns(os.path.join(patterns_path, PATTERN_FILE_SUFFIX))
    
    # Find constellation folders
    constellation_folders = [f for f in target_folder.iterdir() 
                           if f.is_dir() and not f.name.startswith('.')]
    
    constellation_folders = h.sort_by_number_first(constellation_folders)
    
    if config.verbose:
        print(f"Found {len(constellation_folders)} constellation folders")

    all_results = []
    max_patches = 0  # Will be determined dynamically by scanning folders
    
    for i, constellation_folder in enumerate(constellation_folders, 1):
        folder_name = constellation_folder.name
        print(f"Processing folder {i}: {folder_name}")
        try:
            patch_results, constellation_prediction, confidence_score = master.master_function(str(constellation_folder), patterns)
        
        except Exception as e:
            print(f"Error processing folder {folder_name}: {e}")
            patch_results = {}
            constellation_prediction = "unknown"      
        
        # Track maximum patches across all folders
        max_patches = max(max_patches, len(patch_results))

        # Print number of maximum patches
        if config.verbose:
            print(f"Maximum patches so far: {max_patches}")

        # Store results
        result_row = {
            "S.no": i,
            "Folder No.": folder_name,
            "patch_results": patch_results,
            "Constellation prediction": constellation_prediction,
            "Confidence": confidence_score
        }
        all_results.append(result_row)
        if config.verbose:
            print(result_row)

    # Format results for CSV output
    for result in all_results:
        patch_results = result.pop("patch_results")
        sorted_patches = sorted(patch_results.items(), key=lambda x: x[0])
        
        # Add patch columns (patch 1, patch 2, ..., patch N)
        for patch_idx in range(1, max_patches + 1):
            col_name = f"patch {patch_idx}"
            if patch_idx <= len(sorted_patches):
                if sorted_patches[patch_idx - 1][1] != -1:
                    _, (x, y) = sorted_patches[patch_idx - 1]
                    result[col_name] = f"({x}, {y})"
                else:
                    result[col_name] = "-1"
            else:
                result[col_name] = "-1"
        
    # Create DataFrame with proper column order
    df = pd.DataFrame(all_results)
    # Ensure correct column order
    base_cols = ["S.no", "Folder No."]
    patch_cols = [f"patch {i}" for i in range(1, max_patches + 1)]
    final_cols = base_cols + patch_cols + ["Constellation prediction", "Confidence"] 
    
    df = df.reindex(columns=final_cols)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Constellation Classification Assignment")
    parser.add_argument("root_folder", help="Root folder containing data and patterns")
    parser.add_argument("-f", "--folder", required=True,
                       help="Target folder to process (e.g., 'test', 'validation')")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                       help="Enable config.verbose output")
    
    args = parser.parse_args()
    config.verbose = args.verbose  # Set global config.verbose flag in config module
    try:
        # Process the data
        results_df = process_constellation_data(
            args.root_folder,
            args.folder
        )

        # Save results in the same directory as this script
        script_dir = os.path.join(args.root_folder, "submissions", TEAM_FOLDER_NAME)
        output_file = f"{TEAM_FOLDER_NAME}_{args.folder}_results.csv"
        print(f"Saving results to {os.path.join(script_dir, output_file)}...")
        results_df.to_csv(os.path.join(script_dir, output_file), index=False)
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# ==============================================================================
# INPUT/OUTPUT SPECIFICATION
# ==============================================================================

"""
INPUT STRUCTURE:
root_folder/
├── test/                    # Your target folder (or validation/, etc.)
│   ├── constellation_1/
│   │   ├── [sky_image]      # Image file (various names/formats)
│   │   └── patches/         # Subfolder with patch templates
│   │       ├── patch_01.png
│   │       ├── patch_02.png
│   │       └── ...
│   ├── constellation_2/
│   │   └── ...
│   └── ...
├── patterns/                # Reference constellation patterns
│   ├── constellation_name_pattern.png
│   └── ...
└── submissions/your_netid/  # Your script location
    └── your_netid.py

REQUIRED OUTPUT CSV FORMAT:
- Filename: netid_{folder_name}_results.csv (e.g., abc123_test_results.csv)
- Location: Same directory as your script
- Columns: S.no, Folder No., patch 1, patch 2, ..., patch N, Constellation prediction

EXAMPLE OUTPUT:
S.no,Folder No.,patch 1,patch 2,patch 3,patch 4,...,patch N,Constellation prediction
1,constellation_1,(3055,6543),(3895,4611),(4463,4661),-1,...,-1,bootes
2,constellation_2,-1,(2456,3789),-1,-1,...,-1,orion
3,constellation_3,-1,-1,-1,-1,...,-1,unknown

(Note: N = maximum number of patches found across all folders)

COORDINATE FORMAT:
- Successful match: (x,y) - center coordinates of matched patch
- No match/rejected: -1
- Number of patch columns: Dynamically determined by folder with most patches

CONSTELLATION NAMES:
- Lowercase format (e.g., 'bootes', 'orion', 'corona-australis')
- Extract from pattern filenames (remove '_pattern' suffix)

USAGE:
python your_netid.py /path/to/Data_Project1 -f test -v
python your_netid.py /path/to/Data_Project1 -f validation
"""