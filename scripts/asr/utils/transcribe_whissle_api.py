import requests
import argparse
import json # Import json module to handle boosted_lm_words
import math # Import math for exp function

# Function to list available ASR models
def list_asr_models(auth_token):
    url = f'https://api.whissle.ai/v1/list-asr-models?auth_token={auth_token}'  # Pass auth_token as a query parameter
    response = requests.get(url)
    return response.json() if response.status_code == 200 else response.text

# Example usage - Keep this if you still want to list models when running the script directly without transcribe args
# models = list_asr_models(auth_token="d7293d79d38f4394")
# print("Available models:", models)

# Function to transcribe audio using a specific ASR model
def transcribe_audio(auth_token, model_name, audio_path, word_timestamps=None, boosted_lm_words=None, boosted_lm_score=None):
    """
    Transcribes an audio file using the Whissle AI API.

    Args:
        auth_token (str): Your API authentication token.
        model_name (str): The name of the ASR model to use.
        audio_path (str): The path to the audio file.
        word_timestamps (bool, optional): Whether to return word timestamps (required for confidence ratio). Defaults to None.
        boosted_lm_words (list, optional): A list of words to boost in the language model. Defaults to None.
        boosted_lm_score (int, optional): The score to apply for boosted words. Defaults to None.

    Returns:
        dict or str: The JSON response from the API or an error message.
    """
    url = f'https://api.whissle.ai/v1/conversation/STT?model_name={model_name}&auth_token={auth_token}'
    # Use a context manager for opening the file to ensure it's closed
    files = None
    data = {} # Initialize data dictionary for optional form fields

    # Add optional parameters to the data dictionary if they are provided
    if word_timestamps is not None:
        # API expects '1' for true, '0' for false, or simply the presence of the key for true
        data['word_timestamps'] = '1' if word_timestamps else '0'
    if boosted_lm_words is not None:
        # Change from json.dumps to str() to match the literal string format like ['word1', 'word2']
        data['boosted_lm_words'] = str(boosted_lm_words)
    if boosted_lm_score is not None:
        data['boosted_lm_score'] = str(boosted_lm_score) # Ensure score is passed as string

    try:
        # Open the file within the try block using a context manager
        with open(audio_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            response = requests.post(url, files=files, data=data)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
    except FileNotFoundError:
        return f"Error: Audio file not found at {audio_path}"
    except requests.exceptions.RequestException as e:
        # Try to get more details from the response if available
        error_details = ""
        try:
            # Check if response exists and has text content
            if hasattr(e, 'response') and e.response is not None and e.response.text:
                 error_details = f" Server response: {e.response.text}"
        except Exception:
             pass # Ignore errors trying to get more details
        return f"Request failed: {e}{error_details}"
    except json.JSONDecodeError as e:
        # Include the response text that failed to parse
        return f"Failed to decode JSON response: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}"
    # No finally block needed for file closing when using 'with open(...)'


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whissle AI API.")

    # Required arguments
    parser.add_argument("--auth_token", required=True, help="Your API authentication token.")
    parser.add_argument("--model_name", required=True, help="Name of the ASR model to use (e.g., 'en-US-0.6b', 'ru-RU-with-lm').")
    parser.add_argument("--audio_path", required=True, help="Path to the audio file.")

    # Optional arguments
    parser.add_argument("--word_timestamps", action='store_true', help="Enable word timestamps (pass '1').")
    parser.add_argument("--boosted_lm_words", type=str, help="JSON string of words to boost (e.g., '[\"reformer\", \"pipeline\"]'). Note: Internally converted to list string.")
    parser.add_argument("--boosted_lm_score", type=int, help="Score for boosted words (e.g., 80).")
    parser.add_argument("--list_models", action='store_true', help="List available ASR models and exit.")
    # New arguments for confidence ratio
    parser.add_argument("--calculate_confidence_ratio", action='store_true',
                        help="Calculate and report the ratio of words meeting the confidence threshold.")
    parser.add_argument("--confidence_threshold", type=float, default=0.2,
                        help="Probability threshold for word confidence (default: 0.2). Used with --calculate_confidence_ratio.")


    args = parser.parse_args()

    if args.list_models:
        print("Fetching available models...")
        models_list = list_asr_models(args.auth_token)
        print("Available ASR Models:")
        # Ensure models_list is properly formatted before printing
        if isinstance(models_list, (dict, list)):
             print(json.dumps(models_list, indent=2)) # Pretty print the JSON
        else:
             print(f"Could not retrieve or parse model list: {models_list}")
    else:
        # Parse boosted_lm_words from JSON string if provided
        boosted_words = None
        if args.boosted_lm_words:
            try:
                # Still parse as JSON from command line for easy input
                boosted_words = json.loads(args.boosted_lm_words)
                if not isinstance(boosted_words, list):
                    raise ValueError("boosted_lm_words must be a JSON list of strings.")
                # Ensure all items in the list are strings
                if not all(isinstance(item, str) for item in boosted_words):
                     raise ValueError("All items in boosted_lm_words list must be strings.")
            except json.JSONDecodeError:
                parser.error("Invalid JSON format for --boosted_lm_words. Example: '[\"word1\", \"word2\"]'")
            except ValueError as e:
                 parser.error(str(e))


        print(f"Transcribing {args.audio_path} with model {args.model_name}...")

        # Determine if word timestamps need to be requested
        request_word_timestamps = args.word_timestamps or args.calculate_confidence_ratio

        # Pass optional arguments explicitly
        transcription_result = transcribe_audio(
            auth_token=args.auth_token,
            model_name=args.model_name,
            audio_path=args.audio_path,
            word_timestamps=request_word_timestamps, # Request if needed for ratio or explicitly asked
            boosted_lm_words=boosted_words,
            boosted_lm_score=args.boosted_lm_score
        )

        # Calculate confidence ratio if requested and possible
        if args.calculate_confidence_ratio and isinstance(transcription_result, dict) and 'timestamps' in transcription_result:
            confident_words = 0
            total_words = 0
            word_timestamps_list = transcription_result.get('timestamps', [])

            if word_timestamps_list: # Ensure timestamps list is not empty
                for word_data in word_timestamps_list:
                    if 'confidence' in word_data and 'word' in word_data: # Check if confidence and word exist
                         total_words += 1
                         try:
                             # Confidence is often log probability, convert to probability
                             log_prob = float(word_data['confidence'])
                             prob = math.exp(log_prob)
                             if prob >= args.confidence_threshold:
                                 confident_words += 1
                         except (ValueError, TypeError):
                             # Handle cases where confidence is not a valid number
                             print(f"Warning: Could not parse confidence '{word_data['confidence']}' for word '{word_data['word']}'. Skipping.")
                             pass # Or handle as needed

                if total_words > 0:
                    confidence_ratio = confident_words / total_words
                    transcription_result['word_confidence_ratio'] = round(confidence_ratio, 4) # Add ratio to result
                    transcription_result['confidence_threshold'] = args.confidence_threshold # Add threshold used
                    transcription_result['confident_word_count'] = confident_words
                    transcription_result['total_word_count'] = total_words
                else:
                    transcription_result['word_confidence_ratio'] = None # Indicate ratio couldn't be calculated
                    transcription_result['confidence_threshold'] = args.confidence_threshold
            else:
                 print("Warning: --calculate_confidence_ratio requested, but 'timestamps' key is missing or empty in the result.")
                 transcription_result['word_confidence_ratio'] = None # Indicate ratio couldn't be calculated
                 transcription_result['confidence_threshold'] = args.confidence_threshold

        elif args.calculate_confidence_ratio:
             print("Warning: --calculate_confidence_ratio requested, but transcription result is not a dictionary or 'timestamps' key is missing.")
             # Optionally add placeholder keys to indicate the attempt
             if isinstance(transcription_result, dict):
                 transcription_result['word_confidence_ratio'] = None
                 transcription_result['confidence_threshold'] = args.confidence_threshold


        print("\nTranscription Result:")
        if isinstance(transcription_result, dict):
            print(json.dumps(transcription_result, indent=2, ensure_ascii=False)) # Pretty print JSON, handle non-ASCII
        else:
            print(transcription_result) # Print error message

