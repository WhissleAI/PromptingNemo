#!/bin/bash

# Function to make the llm_response_without_file curl request and measure time
make_llm_request() {
  START=$(date +%s%N)
  RESPONSE=$(curl -s -o /dev/null -w "%{time_total}" -X 'POST' \
    'https://related-wildcat-hugely.ngrok-free.app/llm_response_without_file' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    -d 'content=explain%20about%20indo-persian%20architecture&model_name=openai&emotion=sad&url=&searchengine=&system_instruction=you%20are%20an%20personal%20assistant%2C%20answer%20the%20question&role=&conversation_history=')
  END=$(date +%s%N)
  DURATION=$((END - START))
  echo "$DURATION $RESPONSE" # Return the duration in nanoseconds and the curl total time in seconds
}

# Function to make the transcribe-web2 curl request and measure time
make_asr_request() {
  START=$(date +%s%N)
  RESPONSE=$(curl -s -o /dev/null -w "%{time_total}" --location \
    'https://related-wildcat-hugely.ngrok-free.app/transcribe-web2' \
    --header 'Authorization: Bearer random_token' \
    --form 'language_id="EN"' \
    --form 'audio=@"/home/ksingla/workspace/PromptingNemo/applications/voicebot/demo_audio/EN_karan_angry.wav"')
  END=$(date +%s%N)
  DURATION=$((END - START))
  echo "$DURATION $RESPONSE" # Return the duration in nanoseconds and the curl total time in seconds
}

# Function to make the generate_audio curl request and measure time
make_tts_request() {
  START=$(date +%s%N)
  RESPONSE=$(curl -s -o out.wav -w "%{time_total}" -X 'POST' \
    'https://related-wildcat-hugely.ngrok-free.app/generate_audio' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'model_name=xtts' \
    -F 'text_to_convert=This is the text to convert to audio. This is not real life' \
    -F 'output_filename=output.wav')
  END=$(date +%s%N)
  DURATION=$((END - START))
  echo "$DURATION $RESPONSE" # Return the duration in nanoseconds and the curl total time in seconds
}

# Arrays to store request times
llm_request_times=()
asr_request_times=()
tts_request_times=()

# Start time of the script
SCRIPT_START=$(date +%s%N)

# Run the LLM requests in parallel
for i in {1..5}; do
  llm_request_times+=("$(make_llm_request &)")
done

# Run the ASR requests in parallel
for i in {1..5}; do
  asr_request_times+=("$(make_asr_request &)")
done

# Run the TTS requests in parallel
for i in {1..5}; do
  tts_request_times+=("$(make_tts_request &)")
done

# Wait for all background jobs to finish
wait

# End time of the script
SCRIPT_END=$(date +%s%N)
SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))

# Calculate total and average time for LLM requests
total_llm_time=0
total_llm_curl_time=0
for time in "${llm_request_times[@]}"; do
  nanoseconds=$(echo $time | cut -d' ' -f1)
  curl_time=$(echo $time | cut -d' ' -f2)
  total_llm_time=$((total_llm_time + nanoseconds))
  total_llm_curl_time=$(echo "$total_llm_curl_time + $curl_time" | bc)
done

average_llm_time=$((total_llm_time / 5))
average_llm_curl_time=$(echo "$total_llm_curl_time / 5" | bc -l)

# Calculate total and average time for ASR requests
total_asr_time=0
total_asr_curl_time=0
for time in "${asr_request_times[@]}"; do
  nanoseconds=$(echo $time | cut -d' ' -f1)
  curl_time=$(echo $time | cut -d' ' -f2)
  total_asr_time=$((total_asr_time + nanoseconds))
  total_asr_curl_time=$(echo "$total_asr_curl_time + $curl_time" | bc)
done

average_asr_time=$((total_asr_time / 5))
average_asr_curl_time=$(echo "$total_asr_curl_time / 5" | bc -l)

# Calculate total and average time for TTS requests
total_tts_time=0
total_tts_curl_time=0
for time in "${tts_request_times[@]}"; do
  nanoseconds=$(echo $time | cut -d' ' -f1)
  curl_time=$(echo $time | cut -d' ' -f2)
  total_tts_time=$((total_tts_time + nanoseconds))
  total_tts_curl_time=$(echo "$total_tts_curl_time + $curl_time" | bc)
done

average_tts_time=$((total_tts_time / 5))
average_tts_curl_time=$(echo "$total_tts_curl_time / 5" | bc -l)

# Convert times to milliseconds for display
total_llm_time_ms=$(echo "$total_llm_time / 1000000" | bc)
average_llm_time_ms=$(echo "$average_llm_time / 1000000" | bc)
total_asr_time_ms=$(echo "$total_asr_time / 1000000" | bc)
average_asr_time_ms=$(echo "$average_asr_time / 1000000" | bc)
total_tts_time_ms=$(echo "$total_tts_time / 1000000" | bc)
average_tts_time_ms=$(echo "$average_tts_time / 1000000" | bc)
script_duration_ms=$(echo "$SCRIPT_DURATION / 1000000" | bc)

# Print the results
for time in "${llm_request_times[@]}"; do
  nanoseconds=$(echo $time | cut -d' ' -f1)
  echo "LLM Request took $(echo "$nanoseconds / 1000000" | bc) milliseconds"
done

for time in "${asr_request_times[@]}"; do
  nanoseconds=$(echo $time | cut -d' ' -f1)
  echo "ASR Request took $(echo "$nanoseconds / 1000000" | bc) milliseconds"
done

for time in "${tts_request_times[@]}"; do
  nanoseconds=$(echo $time | cut -d' ' -f1)
  echo "TTS Request took $(echo "$nanoseconds / 1000000" | bc) milliseconds"
done

echo "Total time: $script_duration_ms milliseconds"
echo "Average LLM request time: $average_llm_time_ms milliseconds"
echo "Average curl time for LLM requests: $(printf "%.3f" $average_llm_curl_time) seconds"
echo "Total ASR request time: $total_asr_time_ms milliseconds"
echo "Average ASR request time: $average_asr_time_ms milliseconds"
echo "Average curl time for ASR requests: $(printf "%.3f" $average_asr_curl_time) seconds"
echo "Total TTS request time: $total_tts_time_ms milliseconds"
echo "Average TTS request time: $average_tts_time_ms milliseconds"
echo "Average curl time for TTS requests: $(printf "%.3f" $average_tts_curl_time) seconds"

