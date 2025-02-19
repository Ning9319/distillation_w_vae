#!/bin/bash

# Get the target folder as an argument or use the current directory
TARGET_FOLDER="./rawframes"

# Find all subdirectories inside the target folder (but not the target itself)
find "$TARGET_FOLDER" -mindepth 2 -type d | while read -r dir; do
    # Get the parent directory of the current subfolder
    parent_dir=$(dirname "$dir")

    # Move the subfolder to the target folder
    mv "$dir" "$TARGET_FOLDER"
    
    echo "Moved $dir to $TARGET_FOLDER"
done

# Remove empty directories
find "$TARGET_FOLDER" -type d -empty -delete
echo "Removed all empty subfolders."

echo "All nested subfolders have been moved and empty directories deleted."
