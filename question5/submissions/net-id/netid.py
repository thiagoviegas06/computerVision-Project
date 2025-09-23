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
from pathlib import Path

def process_constellation_data(root_folder, target_folder, verbose=False):
    """
    Main processing function - implement your constellation classification here
    
    Args:
        root_folder: Path to root data directory (contains patterns/, test/, etc.)
        target_folder: Folder to process ('test', 'validation', etc.)
        verbose: Whether to print detailed output
    
    Returns:
        pandas.DataFrame: Results in required CSV format
    """
    
    root_path = Path(root_folder)
    target_path = root_path / target_folder
    patterns_path = root_path / "patterns"
    
    if verbose:
        print(f"Root folder: {root_path}")
        print(f"Target folder: {target_path}")
        print(f"Patterns folder: {patterns_path}")
    
    # Find constellation folders
    constellation_folders = [f for f in target_path.iterdir() 
                           if f.is_dir() and not f.name.startswith('.')]
    constellation_folders = sorted(constellation_folders, key=lambda x: x.name)
    
    if verbose:
        print(f"Found {len(constellation_folders)} constellation folders")
    
    # TODO: IMPLEMENT YOUR ALGORITHM HERE
    all_results = []
    max_patches = 0  # Will be determined dynamically by scanning folders
    
    for i, constellation_folder in enumerate(constellation_folders, 1):
        folder_name = constellation_folder.name
        
        # TODO: Your processing logic here
        # Load sky image, find patches, do template matching, classify constellation
        
        # Example patch results - replace with your algorithm
        patch_results = {
            # Example format:
            # "patch_01.png": (3055, 6543),  # (x, y) coordinates
            # "patch_02.png": (-1, -1),      # No match found
            # Add your actual results here
        }
        
        # TODO: Replace with your constellation classification
        constellation_prediction = "unknown"
        
        # Track maximum patches across all folders
        max_patches = max(max_patches, len(patch_results))
        
        # Store results
        result_row = {
            "S.no": i,
            "Folder No.": folder_name,
            "patch_results": patch_results,
            "Constellation prediction": constellation_prediction
        }
        all_results.append(result_row)
    
    # Format results for CSV output
    for result in all_results:
        patch_results = result.pop("patch_results")
        sorted_patches = sorted(patch_results.items(), key=lambda x: x[0])
        
        # Add patch columns (patch 1, patch 2, ..., patch N)
        for patch_idx in range(1, max_patches + 1):
            col_name = f"patch {patch_idx}"
            
            if patch_idx <= len(sorted_patches):
                patch_name, (x, y) = sorted_patches[patch_idx - 1]
                if x == -1 and y == -1:
                    result[col_name] = "-1"
                else:
                    result[col_name] = f"({x},{y})"
            else:
                result[col_name] = "-1"
    
    # Create DataFrame with proper column order
    df = pd.DataFrame(all_results)
    
    # Ensure correct column order
    base_cols = ["S.no", "Folder No."]
    patch_cols = [f"patch {i}" for i in range(1, max_patches + 1)]
    final_cols = base_cols + patch_cols + ["Constellation prediction"]
    
    df = df.reindex(columns=final_cols)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Constellation Classification Assignment")
    parser.add_argument("root_folder", help="Root folder containing data and patterns")
    parser.add_argument("-f", "--folder", required=True,
                       help="Target folder to process (e.g., 'test', 'validation')")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Process the data
        results_df = process_constellation_data(
            args.root_folder,
            args.folder,
            args.verbose
        )
        
        # Save results in the same directory as this script
        script_dir = Path(__file__).parent
        output_file = script_dir / f"{args.folder}_results.csv"
        results_df.to_csv(output_file, index=False)
        
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