#!/bin/bash

# Check for the correct number of command-line arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 source_directory destination_directory"
    exit 1
fi

# Source directory
source_dir="$1"

# Destination directory
destination_dir="$2"

# Check if the source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Source directory '$source_dir' does not exist."
    exit 1
fi

# Check if the destination directory exists
if [ ! -d "$destination_dir" ]; then
    echo "Destination directory '$destination_dir' does not exist."
    exit 1
fi

# Change to the source directory
cd "$source_dir" || exit 1

# Loop through files in the source directory
for file in *; do
    if [ -e "$file" ]; then  # Check if the file exists
        mv "$file" "$destination_dir"  # Move the file to the destination directory
    fi
done

echo "Files moved successfully from '$source_dir' to '$destination_dir'."
