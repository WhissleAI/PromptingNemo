#!/bin/bash

# Directory containing the MP3 files
INPUT_DIRECTORY=$1
# Directory where WAV files will be saved
OUTPUT_DIRECTORY=$2

# Create output directory if it does not exist
mkdir -p "$OUTPUT_DIRECTORY"

# Change to the directory with MP3 files
cd "$INPUT_DIRECTORY"

# Loop through all mp3 files and convert them to WAV 16kHz
for file in *.mp3; do
    # Get the base name without extension
    base_name=$(basename "$file" .mp3)
    # Set output file path
    output="$OUTPUT_DIRECTORY/${base_name}.wav"
    # Convert using FFmpeg
    ffmpeg -i "$file" -acodec pcm_s16le -ar 16000 "$output"
done

