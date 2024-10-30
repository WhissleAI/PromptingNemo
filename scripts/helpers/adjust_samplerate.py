#!/bin/bash

# Ensure correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_directory output_directory"
    exit 1
fi

input_dir="$1"
output_dir="$2"

# Ensure the output directory exists, create if not
mkdir -p "$output_dir"

# Function to convert sample rate for a single file
convert_sample_rate() {
    file="$1"
    filename="${file##*/}"  # Extract filename without path
    local output_dir_local="$2"  # Assign the second argument (output_dir) to a local variable
    output_file="$output_dir_local/$filename"  # Use the local variable for output directory
    # Convert the sample rate to 16kHz using sox
    sox "$file" -r 16000 "$output_file"
    rm "$file"
}

export -f convert_sample_rate  # Export the function

# Run conversion in parallel for all files in the input directory using 4 threads
find "$input_dir" -type f -name "*.wav" | parallel -j8 convert_sample_rate {} "$output_dir"
