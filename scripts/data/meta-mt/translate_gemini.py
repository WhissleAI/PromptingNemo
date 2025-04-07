import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from a .env file, if present

# Configure PaLM with your API key
genai.configure(api_key="AIzaSyCVJi5lluRBuxIgSF9VQEfNJw3cOyqd4JA")

def translate_text_with_tags(text, target_language="it"):
    """
    Translates an English text into the specified target language (e.g., Italian),
    preserving all capitalized tag tokens exactly and using them as context
    for better translation quality.

    Args:
        text (str): The original English text that may contain tags like AGE_30_45, GER_MALE, etc.
        target_language (str): The target language for translation (default 'it' for Italian).

    Returns:
        str: The text translated into the target language, preserving the tags.
    """

    # Construct a translator prompt that explains how to handle the tags.
    # We instruct the LLM to:
    #   - keep the capitalized tags EXACTLY as-is,
    #   - treat them as contextual hints about the speaker or situation.
    prompt = f"""
You are an expert translator. Translate the following text from English into {target_language}.
The text contains capitalized tokens like AGE_30_45, GER_MALE, EMOTION_HAP, INTENT_INFORM, etc.
These tokens are additional metadata and must remain EXACTLY as they appear in the output.
They should not be translated or altered in any way.

Text to translate:
{text}
"""

    try:
        # Generate a single response
        model = genai.GenerativeModel("gemini-2.0-flash")  # Or whichever PaLM model is available
        response = model.generate_content(prompt)
        translated_text = response.text.strip()
        return translated_text
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        # Fallback: return original text in case of error
        return text


def process_json_file_for_translation(input_path, output_path,
                                      target_language="it", batch_size=10):
    """
    Reads an input JSON file line by line, translates the 'text' field of each record
    into the specified target language (preserving tags), and writes out a JSON file
    with the translated text.

    Args:
        input_path (str): Path to the input JSON file (one record per line).
        output_path (str): Path to output the translated JSON lines.
        target_language (str): The language code to translate the text into.
        batch_size (int): Number of lines to send to the model at once (optional).
    """
    # Ensure output directory exists
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Remove existing output file if it exists
    #if os.path.exists(output_path):
    #    os.remove(output_path)

    # Read all lines of JSON
    with open(input_path, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    total_records = len(lines)
    print(f"Total records to process: {total_records}")

    # Process in batches
    for i in range(0, total_records, batch_size):
        batch_lines = lines[i : i + batch_size]
        batch_records = []
        for line in batch_lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                batch_records.append(record)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON line: {line}\nError: {e}")

        print(f"\nProcessing batch {i // batch_size + 1} / {(total_records + batch_size - 1) // batch_size}...")

        # Collect the text to be translated
        texts_to_translate = []
        for record in batch_records:
            texts_to_translate.append(record.get('text', ""))

        # Translate each text in the batch individually.
        # (If you need a single multi-text prompt, you can do that too.)
        translated_texts = []
        for text in texts_to_translate:
            translated_text = translate_text_with_tags(text, target_language=target_language)
            print(translated_text)
            translated_texts.append(translated_text)

        # Write results to the output file
        with open(output_path, 'a', encoding='utf-8') as f_out:
            for record, trans_text in zip(batch_records, translated_texts):
                # Replace the original text with the translation
                record['text'] = trans_text
                # Write the updated record as JSON
                out_str = json.dumps(record, ensure_ascii=False)
                f_out.write(out_str + '\n')

        print(f"Batch {i // batch_size + 1} processed and saved.")

    print(f"\nTranslation complete. Output saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    input_json_path = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/people_speech_deduplicated.jsonl"
    output_json_path = "output.json"
    target_language = "it"  # e.g., "it" for Italian, "es" for Spanish, etc.

    process_json_file_for_translation(
        input_path=input_json_path,
        output_path=output_json_path,
        target_language=target_language,
        batch_size=10
    )

