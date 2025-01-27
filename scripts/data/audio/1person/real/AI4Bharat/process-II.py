# Import necessary libraries
import os
import json
import openai
import time
import getpass
import csv
import re

# Path to the JSONL file
jsonl_file = '/external1/datasets/indicVoices_v3/hindi/Hindi_processed/Hindi_manifest_valid.jsonl'  # Update this path as needed

# Path to save the intermediate JSON file
intermediate_json_path = '/external1/datasets/indicVoices_v3/hindi/Hindi_processed/intermediate_data.json'  # Update this path as needed

output_json_path = '/external1/datasets/indicVoices_v3/hindi/Hindi_processed/annotated_data.json'  # Update this path as needed
output_csv_path = '/external1/datasets/indicVoices_v3/hindi/Hindi_processed/annotated_data.csv'  # Update this path as needed

# Ensure directories for the output files exist


openai.api_key =''

# Paths to save the output files

# Collect all records from the JSONL file
records = []

# Read the JSONL file
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        if not line.strip():  # Skip empty lines
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            continue
        # Check if all required keys exist
        required_keys = ['path', 'text', 'intent', 'age_group', 'gender', 'dialect']
        if all(key in data for key in required_keys):
            # Clean the text by removing annotations like [inhaling], [uhh], etc.
            cleaned_text = re.sub(r'\[.*?\]', '', data['text']).strip()
            # Replace multiple spaces with a single space
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            records.append({
                'path': data['path'],
                'text': cleaned_text,
                'intent': data['intent'],
                'age_group': data['age_group'],
                'gender': data['gender'],
                'dialect': data['dialect']
            })
        else:
            print(f"Missing required keys in data: {data}")

# Save the records to the intermediate JSON file
os.makedirs(os.path.dirname(intermediate_json_path), exist_ok=True)
with open(intermediate_json_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=4)

print(f"Data extraction complete. Intermediate data saved to '{intermediate_json_path}'.")


with open(intermediate_json_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

# Initialize annotated_records
annotated_records = []

# Check if output JSON file already exists and load it to resume from last point
if os.path.exists(output_json_path):
    with open(output_json_path, 'r', encoding='utf-8') as f_json:
        annotated_records = json.load(f_json)
    print(f"Loaded {len(annotated_records)} existing annotated records.")
else:
    print("No existing annotated records found. Starting fresh.")

# Create a set of paths for already annotated records to avoid duplicates
annotated_paths = set(record['path'] for record in annotated_records)

# Total number of records
total_records = len(records)
print(f"Total sentences to process: {total_records}")

# Process records in batches
batch_size = 10  # Set batch size to 10 as per your request
total_batches = (total_records + batch_size - 1) // batch_size

print(f"Processing in batches of {batch_size} sentences.")

for batch_num in range(total_batches):
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, total_records)
    batch_records = records[start_idx:end_idx]
    print(f"\nProcessing batch {batch_num + 1}/{total_batches} (sentences {start_idx + 1} to {end_idx})...")

    # Prepare the batch of sentences to annotate
    sentences_to_annotate = []
    paths = []
    intents = []
    age_groups = []
    genders = []
    dialects = []

    for idx, record in enumerate(batch_records, start=start_idx + 1):
        path = record['path']

        # Skip records that have already been annotated
        if path in annotated_paths:
            print(f"Skipping already annotated sentence {idx}/{total_records}: {path}")
            continue

        print(f"Preparing sentence {idx}/{total_records}: {path}")
        sentence = record['text']
        intent = record['intent']
        age_group = record['age_group']
        gender = record['gender']
        dialect = record['dialect']

        sentences_to_annotate.append(sentence)
        paths.append(path)
        intents.append(intent)
        age_groups.append(age_group)
        genders.append(gender)
        dialects.append(dialect)

    if not sentences_to_annotate:
        continue  # Skip if all sentences in this batch are already annotated

    def annotate_sentences(sentences):
        # Prepare the prompt with multiple sentences
        prompt = f'''
Given a list of sentences in Hindi, annotate each sentence individually with the appropriate entity tags from the provided list. The sentences may relate to various actions such as managing tasks, controlling devices, sending notifications, scheduling events, updating information, or offering assistance.

Instructions:

    Annotate each sentence separately.
    Use the entity tags to indicate the start and end of each entity phrase in the sentence.
    The tagging format is ENTITY_<type> to start an entity and there must be an END to close it.
    Any spaces in the ENTITY_<type> must be replaced with _.
    Only use the entity types provided in the list.
    Strictly remove any text that appears between < and > (including the angle brackets themselves) always
    Do not add any additional text or explanations.
    Ensure the output is a JSON array containing only the annotated sentences in the same order as the input sentences, without any markdown or formatting.

Entities:

[ "PERSON NAME", "ORGANIZATION", "LOCATION", "DATE", "TIME", "DURATION", "EMAIL", "PHONE NUMBER", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP CODE", "CURRENCY", "PRICE", "PRODUCT", "SERVICE", "BRAND", "EVENT", "PERCENTAGE", "AGE", "TEMPERATURE", "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "SPEED", "LANGUAGE", "NATIONALITY", "RELIGION", "JOB TITLE", "COMPANY NAME", "DEVICE NAME", "OPERATING SYSTEM", "SOFTWARE VERSION", "COLOR", "SHAPE", "MATERIAL", "MODEL NUMBER", "LICENSE PLATE", "VEHICLE MAKE", "VEHICLE MODEL", "VEHICLE TYPE", "FLIGHT NUMBER", "HOTEL NAME", "BOOKING REFERENCE", "PAYMENT METHOD", "CREDIT CARD NUMBER", "ACCOUNT NUMBER", "INSURANCE PROVIDER", "POLICY NUMBER", "BANK NAME", "TAX ID", "SOCIAL SECURITY NUMBER", "DRIVER'S LICENSE", "PASSPORT NUMBER", "WEBSITE", "URL", "IP ADDRESS", "MAC ADDRESS", "USERNAME", "PASSWORD", "FOOD ITEM", "DRINK ITEM", "CUISINE", "INGREDIENT", "DISH NAME", "MENU ITEM", "ORDER NUMBER", "PAYMENT AMOUNT", "DELIVERY TIME", "DELIVERY DATE", "APPOINTMENT DATE", "APPOINTMENT TIME", "ROOM NUMBER", "HOSPITAL NAME", "DOCTOR NAME", "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST NAME", "TEST RESULT", "INSURANCE PLAN", "CLAIM NUMBER", "POLICY HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY CONTACT", "PROJECT NAME", "TASK", "MEETING", "AGENDA", "ACTION ITEM", "DEADLINE", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE" ]

Example:

Input Sentences:
[
    "हाँ जी मैं अनीता बात कर रही हूँ आज ही मैने आपके एजेंसी से जो कैब है उसकी सुविधा प्राप्त की थी मैं उस सुविधा से बहुत ज़्यादा असंतुष्ट हूँ बहुत ज़्यादा निराश हूँ और मैं उसी के लिए आपसे शिकायत करना चाहती हूँ",
    "राजस्थान का अन्य राज्यों से सा संबंध बहोत अच्छा रहा है और राजस्थान खनिजों का अजेबघर कहा जाता है तो राजस्थान के अंदर मसाले खनिज बहोत ज्यादा तादात मे मिलते है तो इनका",
    "एक समय की बात है मैं किसी कंपनी में नौकरी कर रहा था और मैं अपने काम को बड़ी अच्छी तरह से अपना काम पूरा करता था लेकिन उसके बाद भी",
    "चेक करो कि क्या आईआरसीटीसी से पचास हज़ार रुपये का रिफ़ंड प्रोसेस हुआ है या नहीं",
    "हिंदी भाषा हमारे पुरे इतिहास में अन्य भाषाओं संस्कृतियों से बहोत प्रभावित हुई है जैसे कि हिंदी पर बहोत ही विदेशी और स्वदेशी भाषाओं का"
]

Tagged Output:
[
    "हाँ जी मैं ENTITY_PERSON_NAME अनीता END बात कर रही हूँ आज ही मैंने आपके ENTITY_ORGANIZATION एजेंसी END से जो ENTITY_VEHICLE_TYPE कैब END है उसकी ENTITY_SERVICE सुविधा END प्राप्त की थी मैं उस ENTITY_SERVICE सुविधा END से बहुत ज़्यादा असंतुष्ट हूँ बहुत ज़्यादा निराश हूँ और मैं उसी के लिए आपसे ENTITY_COMPLAINT शिकायत END करना चाहती हूँ",
    "ENTITY_STATE राजस्थान END का ENTITY_STATE अन्य राज्यों END से संबंध बहोत अच्छा रहा है और ENTITY_STATE राजस्थान END ENTITY_PRODUCT खनिजों END का अजेबघर कहा जाता है तो ENTITY_STATE राजस्थान END के अंदर ENTITY_PRODUCT मसाले END ENTITY_PRODUCT खनिज END बहोत ज्यादा तादात में मिलते है तो इनका",
    "एक समय की बात है मैं किसी ENTITY_COMPANY_NAME कंपनी END में ENTITY_JOB_TITLE नौकरी END कर रहा था और मैं अपने ENTITY_TASK काम END को बड़ी अच्छी तरह से अपना ENTITY_TASK काम END पूरा करता था लेकिन उसके बाद भी",
    "चेक करो कि क्या ENTITY_ORGANIZATION आईआरसीटीसी END से ENTITY_PRICE पचास हज़ार रुपये END का ENTITY_ACTION रिफ़ंड प्रोसेस END हुआ है या नहीं ",
    "ENTITY_LANGUAGE हिंदी END भाषा हमारे पुरे इतिहास में ENTITY_LANGUAGE अन्य भाषाओं END ENTITY_LANGUAGE संस्कृतियों END से बहोत प्रभावित हुई है जैसे कि ENTITY_LANGUAGE हिंदी END पर बहोत ही ENTITY_NATIONALITY विदेशी END और ENTITY_NATIONALITY स्वदेशी END भाषाओं का"

]

Sentences to Annotate:
{json.dumps(sentences, ensure_ascii=False)}
'''
        try:
            #client = openai.OpenAI(api_key='')
            response = openai.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )
                # Get the assistant's reply
            assistant_reply = dict(response).get('choices')[0].message.content
            try:
                annotated_sentences = json.loads(assistant_reply)
                if isinstance(annotated_sentences, list):
                    return annotated_sentences
                else:
                    print("Assistant did not return a list. Fallback to raw reply.")
                    return [assistant_reply] * len(sentences)  # Fallback
            except json.JSONDecodeError:
                print("JSON decoding failed. Fallback to raw replies.")
                return [assistant_reply] * len(sentences)  # Fallback
        except Exception as e:
            print(f"Error annotating batch starting at sentence {start_idx + 1}: {e}")
            return sentences  # Return original sentences if annotation fails

    annotated_batch_sentences = annotate_sentences(sentences_to_annotate)

    # Ensure we have annotations for all sentences
    if len(annotated_batch_sentences) != len(sentences_to_annotate):
        print("Mismatch in number of annotated sentences. Using original sentences.")
        annotated_batch_sentences = sentences_to_annotate

    # Save annotations
    for i in range(len(annotated_batch_sentences)):
        annotated_sentence = annotated_batch_sentences[i]
        path = paths[i]
        intent = intents[i]
        age_group = age_groups[i]
        gender = genders[i]
        dialect = dialects[i]

        # Construct the final output
        def format_tag(tag_type, value):
            return f"{tag_type}_{value.upper().replace(' ', '_').replace('-', '_').replace('/', '_')}"

        final_output = f"{annotated_sentence} {format_tag('INTENT', intent)} {format_tag('AGE', age_group)} {format_tag('GENDER', gender)} {format_tag('DIALECT', dialect)}"

        annotated_records.append({'path': path, 'Final Output': final_output})

        # Update the set of annotated paths
        annotated_paths.add(path)

    # Save progress after each batch
    # Write to JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f_json:
        json.dump(annotated_records, f_json, ensure_ascii=False, indent=4)

    # Write to CSV file
    csv_columns = ['path', 'Final Output']
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(annotated_records)

    # Optional: Sleep to respect rate limits
    # time.sleep(1)  # Adjust or remove as needed

print("\nAnnotation complete.")
print(f"Files saved as '{os.path.basename(output_json_path)}' and '{os.path.basename(output_csv_path)}'.")