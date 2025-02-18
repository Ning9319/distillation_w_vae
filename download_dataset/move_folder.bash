#!/bin/bash

# Define the main folder (change '.' to the absolute path if needed)
MAIN_FOLDER="."

# Find all sub-subfolders and move them to the main folder
find "$MAIN_FOLDER" -mindepth 2 -type d -exec mv -t "$MAIN_FOLDER" {} +

# Remove empty subdirectories
find "$MAIN_FOLDER" -mindepth 1 -type d -empty -delete

echo "All sub-subfolders have been moved to the main folder, and empty directories deleted."