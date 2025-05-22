#!/bin/bash

# Base directory containing language folders
BASE_DIR="/datasets/cv/cv-corpus-15.0-2023-09-08"

# Function to process a single language
process_language() {
    local lang_dir="$1"
    local lang=$(basename "$lang_dir")
    echo "Processing language: $lang"
    
    # Create clips_wav directory if it doesn't exist
    clips_wav_dir="${lang_dir}clips_wav"
    mkdir -p "$clips_wav_dir"
    
    # Convert each MP3 file to WAV
    for mp3_file in "${lang_dir}clips"/*.mp3; do
        if [ -f "$mp3_file" ]; then
            filename=$(basename "$mp3_file")
            wav_filename="${filename%.mp3}.wav"
            wav_path="${clips_wav_dir}/${wav_filename}"
            
            # Skip if WAV file already exists
            if [ -f "$wav_path" ]; then
                echo "WAV file already exists, skipping: $filename"
                rm "$mp3_file"
                continue
            fi
            
            echo "Converting: $filename"
            ffmpeg -i "$mp3_file" -ar 16000 -ac 1 -c:a pcm_s16le "$wav_path"
            
            # Check if conversion was successful
            if [ $? -eq 0 ]; then
                echo "Successfully converted: $filename"
                rm "$mp3_file"
                echo "Deleted original MP3: $filename"
            else
                echo "Error converting: $filename"
            fi
        fi
    done
    echo "Completed processing language: $lang"
}

# Export the function so it can be used by parallel
export -f process_language

# Process all languages in parallel
find "$BASE_DIR" -maxdepth 1 -type d -not -path "$BASE_DIR" | parallel process_language

echo "All conversions completed!" 