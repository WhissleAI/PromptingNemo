#!/bin/bash

# Function to make the curl request and measure time
make_request() {
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

# Array to store request times
request_times=()
curl_times=()

# Start time of the script
SCRIPT_START=$(date +%s%N)

# Run the requests in parallel
for i in {1..5}; do
  request_times+=("$(make_request &)")
done

# Wait for all background jobs to finish
wait

# End time of the script
SCRIPT_END=$(date +%s%N)
SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))

# Calculate total and average time
total_time=0
total_curl_time=0
for time in "${request_times[@]}"; do
  nanoseconds=$(echo $time | cut -d' ' -f1)
  curl_time=$(echo $time | cut -d' ' -f2)
  total_time=$((total_time + nanoseconds))
  total_curl_time=$(echo "$total_curl_time + $curl_time" | bc)
done

average_time=$((total_time / 5))
average_curl_time=$(echo "$total_curl_time / 5" | bc -l)

# Convert times to milliseconds for display
total_time_ms=$(echo "$total_time / 1000000" | bc)
average_time_ms=$(echo "$average_time / 1000000" | bc)
script_duration_ms=$(echo "$SCRIPT_DURATION / 1000000" | bc)

# Print the results
for time in "${request_times[@]}"; do
  nanoseconds=$(echo $time | cut -d' ' -f1)
  echo "Request took $(echo "$nanoseconds / 1000000" | bc) milliseconds"
done
echo "Total time: $script_duration_ms milliseconds"
echo "Average time: $average_time_ms milliseconds"
echo "Average curl time: $(printf "%.3f" $average_curl_time) seconds"

