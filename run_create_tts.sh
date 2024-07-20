#!/bin/bash

# Array of language codes
language_codes=("EN" "ES" "FR" "DE" "HI" "PA" "BN" "MR" "GU" "TE")

# Function to run the python script for a given language
run_script() {
  lang=$1
  python scripts/data/audio/synthetic/create_tts_manifest_xtts.py $lang 1
}

# Loop through each language code and run the python script in parallel (two at a time)
for ((i = 0; i < ${#language_codes[@]}; i+=2)); do
  # Run the script for the current language code in the background
  run_script "${language_codes[$i]}" &
  if ((i+1 < ${#language_codes[@]})); then
    # Run the script for the next language code in the background if it exists
    run_script "${language_codes[$i+1]}" &
  fi
  # Wait for both background processes to finish
  wait
done
