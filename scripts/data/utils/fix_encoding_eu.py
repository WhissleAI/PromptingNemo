import argparse
import json

def fix_manifest(input_path, output_path):
    """
    Reads a manifest file with one JSON object per line, decodes Unicode escapes
    into actual characters, skips invalid lines, and writes the cleaned JSON
    objects to a new file with proper Unicode encoding.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                # Parse JSON (this automatically decodes \u escapes)
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

            # Dump JSON with ensure_ascii=False to keep actual Unicode characters
            json.dump(record, outfile, ensure_ascii=False)
            outfile.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Fix manifest JSON encoding")
    parser.add_argument(
        "input_manifest",
        help="Path to the input manifest file (one JSON object per line)."
    )
    parser.add_argument(
        "output_manifest",
        help="Path where the fixed manifest will be written."
    )
    args = parser.parse_args()

    fix_manifest(args.input_manifest, args.output_manifest)
    print(f"Fixed manifest written to {args.output_manifest}")

if __name__ == "__main__":
    main()

# Usage example:
# python fix_manifest.py samples_manifest.json fixed_manifest.json

