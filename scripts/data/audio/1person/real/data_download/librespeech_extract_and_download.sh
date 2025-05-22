#!/bin/bash

# Function to download and extract a dataset
download_and_extract() {
    local lang=$1
    local url="https://dl.fbaipublicfiles.com/mls/mls_${lang}.tar.gz"
    local file="mls_${lang}.tar.gz"
    
    echo "Starting download of ${lang} dataset..."
    wget "$url" && {
        echo "Extracting ${lang} dataset..."
        tar -xzf "$file" && rm "$file"
        echo "${lang} dataset completed!"
    }
}

# Start all downloads and extractions in parallel
download_and_extract "german" &
download_and_extract "french" &
download_and_extract "spanish" &
download_and_extract "italian" &
download_and_extract "portuguese" &
download_and_extract "polish" &

# Wait for all background processes to complete
echo "All downloads and extractions started in parallel..."
wait

echo "All datasets have been downloaded and extracted successfully!"
