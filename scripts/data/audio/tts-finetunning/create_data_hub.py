import json

# Map from your emotion labels to Orpheus tags
EMOTION_TAGS = {
    "sad": "<sigh>",
    "hap": "<laugh>",
    "ang": "<groan>",
    "neu": ""  # keep neutral plain
}

def add_emotion_tag(record):
    tag = EMOTION_TAGS.get(record.get("emotion", "").lower(), "")
    text = record.get("transcription", "").strip()
    if tag:
        # Prepend tag and ensure a space
        record["transcription"] = f"{tag} {text}"
    else:
        record["transcription"] = text
    return record

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line)
            data = add_emotion_tag(data)
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_jsonl = "E:\Meta_asr\datasets\data\output.jsonl"  # Replace with your actual input file path
    output_jsonl = "E:\Meta_asr\datasets\data\output_hub.jsonl"  # Output file path
    process_jsonl(input_jsonl, output_jsonl)
    
    print("✅ New file saved to input_with_tags.jsonl")
