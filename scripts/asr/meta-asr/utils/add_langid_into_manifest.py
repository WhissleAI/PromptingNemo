import argparse
import json
import re

import tqdm
from lingua import Language, LanguageDetectorBuilder


def remove_tags(text):
    """Removes tags which are uppercase words, optionally with numbers and underscores."""
    # This regex matches words that consist of uppercase letters, numbers, and underscores.
    # These are assumed to be tags.
    no_tags = re.sub(r'\b[A-Z0-9_]+\b', '', text)
    # Clean up extra spaces
    cleaned_text = ' '.join(no_tags.split())
    return cleaned_text


def main():
    parser = argparse.ArgumentParser(description="Add language ID to a NeMo manifest file using text.")
    parser.add_argument("--input_manifest", required=True, type=str, help="Path to the input manifest file.")
    parser.add_argument("--output_manifest", required=True, type=str, help="Path to the output manifest file.")
    args = parser.parse_args()

    print("Loading language identification model...")
    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    print("Model loaded.")

    with open(args.input_manifest, 'r', encoding='utf-8') as infile, \
         open(args.output_manifest, 'w', encoding='utf-8') as outfile:

        lines = infile.readlines()

        for line in tqdm.tqdm(lines, desc="Processing manifest"):
            try:
                item = json.loads(line)
                text = item.get('text')
                
                if text:
                    cleaned_text = remove_tags(text)
                    language = detector.detect_language_of(cleaned_text)
                    if language:
                        language_id = language.iso_code_639_1.name
                        item['language_id'] = language_id
                else:
                    print(f"Skipping line, no text field: {line.strip()}")

                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print(f"Skipping line due to JSONDecodeError: {line.strip()}")
            except Exception as e:
                print(f"An error occurred on line: {line.strip()}. Error: {e}")

    print(f"Finished processing. Output written to {args.output_manifest}")

if __name__ == '__main__':
    main()
