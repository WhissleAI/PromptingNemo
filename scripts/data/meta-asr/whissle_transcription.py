import os
import json
import asyncio
from whissle import WhissleClient

def parse_word_boost_file(file_path):
    """Parse a text file containing word boost lists and return them as a single list."""
    boosted_words = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        try:
            # Try to parse as a list
            words = eval(line)
            if isinstance(words, list):
                boosted_words.extend(words)
        except:
            # If not parseable as a list, add as a single word
            boosted_words.append(line.strip())
    
    return boosted_words

async def transcribe_with_whissle(audio_path, model_name="en-US-0.6b", boosted_lm_words=None, boosted_lm_score=80):
    """
    Transcribe audio using Whissle.

    Args:
        audio_path (str): Path to the audio file.
        model_name (str): The model name to use for transcription.
        boosted_lm_words (list): List of words to boost in the transcription.
        boosted_lm_score (int): Score for boosted words.

    Returns:
        dict: The transcription response.
    """
    try:
        whissle = WhissleClient(auth_token='5fa8b1dfb042419e')
        
        # Add the boosted_lm_words parameter only if it's provided and not empty
        kwargs = {
            "model_name": model_name,
            "timestamps": True,
        }
        
        if boosted_lm_words and len(boosted_lm_words) > 0:
            kwargs["boosted_lm_words"] = boosted_lm_words
            kwargs["boosted_lm_score"] = boosted_lm_score
            
        response = await whissle.async_client.speech_to_text(audio_path, **kwargs)
        return response, None
    except Exception as e:
        return None, str(e)

async def transcribe_directory_async(audio_dir, word_boost_file=None, boosted_lm_score=80, model_name="en-US-0.6b"):
    """Transcribe all .wav files in a directory asynchronously."""
    # Get word boost list if provided
    boosted_lm_words = []
    if word_boost_file and os.path.exists(word_boost_file):
        boosted_lm_words = parse_word_boost_file(word_boost_file)
        print(f"Loaded {len(boosted_lm_words)} boosted words")
    
    # Collect all .wav files from the directory
    wav_files = [
        os.path.join(audio_dir, file_name)
        for file_name in os.listdir(audio_dir)
        if file_name.lower().endswith('.wav')
    ]
    
    # Process files one by one (to avoid overwhelming the API)
    results = {}
    for audio_path in wav_files:
        file_name = os.path.basename(audio_path)
        print(f"Transcribing {file_name}...")
        
        response, error = await transcribe_with_whissle(
            audio_path, 
            model_name,
            boosted_lm_words,
            boosted_lm_score
        )
        
        if response:
            results[file_name] = response
            print(f"Transcribed: {response['text']}")
        else:
            print(f"Error transcribing {file_name}: {error}")
        
        # Save results after each file in case of later failures
        with open("transcription_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    return results


async def main():
    # Configuration
    audio_directory = "/external2/datasets/hf_data_output_audio"
    word_boost_file = "/hydra2-prev/home/compute/workspace_himanshu/Data-store/english/hf_data_jsonl/keyword_data.txt"
    model_name = "en-US-0.6b"
    
    
    results = await transcribe_directory_async(
        audio_directory,
        word_boost_file=word_boost_file,
        boosted_lm_score=80,
        model_name=model_name
    )
    
    print(f"Transcription complete. Processed {len(results)} files.")

if __name__ == "__main__":
    asyncio.run(main())