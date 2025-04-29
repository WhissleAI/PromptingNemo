#!/bin/bash

# Check if input file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file.jsonl>"
    exit 1
fi

input_file="$1"
total_lines=$(wc -l < "$input_file")

# Calculate split points
train_lines=$((total_lines * 90 / 100))
remaining_lines=$((total_lines - train_lines))
test_lines=$((remaining_lines / 2))
valid_lines=$((remaining_lines - test_lines))

# Create train file (first 90%)
head -n "$train_lines" "$input_file" > train_librispeech-sp.json
sed -i 's|/external2/datasets/librespeech/mls_spanish_opus|/mls_spanish_opus|g' train_librispeech-sp.json

# Create test file (next 5%)
tail -n "$remaining_lines" "$input_file" | head -n "$test_lines" > test_librispeech_sp.json
sed -i 's|/external2/datasets/librespeech/mls_spanish_opus|/mls_spanish_opus|g' test_librispeech_sp.json

# Create validation file (last 5%)
tail -n "$valid_lines" "$input_file" > valid_librispeech_sp.json
sed -i 's|/external2/datasets/librespeech/mls_spanish_opus|/mls_spanish_opus|g' valid_librispeech_sp.json

echo "Files created successfully:"
echo "1. train_librispeech-sp.json ($train_lines lines)"
echo "2. test_librispeech_sp.json ($test_lines lines)"
echo "3. valid_librispeech_sp.json ($valid_lines lines)" 
