import os
import json
import openai
import time

# Path to the intermediate JSON file
data_path = '/external1/hkoduri/data/malayalam_processed/'
intermediate_json_path = os.path.join(data_path, 'intermediate_data.json')
job_path = os.path.join(data_path, 'job_data.jsonl')
output_json_path = os.path.join(data_path, 'annotated_data.json')
output_csv_path = os.path.join(data_path, 'annotated_data.csv')

# Load records from intermediate JSON
with open(intermediate_json_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

# Load existing annotated records if any
annotated_records = []
if os.path.exists(output_json_path):
    with open(output_json_path, 'r', encoding='utf-8') as f_json:
        annotated_records = json.load(f_json)
    print(f"Loaded {len(annotated_records)} existing annotated records.")
else:
    print("No existing annotated records found. Starting fresh.")

annotated_paths = set(record['path'] for record in annotated_records)
total_records = len(records)
batch_size = 10

print(f"Total sentences to process: {total_records}")
print(f"Processing in batches of {batch_size} sentences.")

def write_job_request(sentences, idx, job_file):
    prompt = f'''
Given a list of sentences in Malayalam, annotate each sentence individually with the appropriate entity tags from the provided list. The sentences may relate to various actions such as managing tasks, controlling devices, sending notifications, scheduling events, updating information, or offering assistance.

Instructions:

    Annotate each sentence separately.
    Use the entity tags to indicate the start and end of each entity phrase in the sentence.
    The tagging format is ENTITY_<type> to start an entity and there must be an END to close it.
    Any spaces in the ENTITY_<type> must be replaced with _.
    Only use the entity types provided in the list.
    Do not add any additional text or explanations.
    Ensure the output is a JSON array containing only the annotated sentences in the same order as the input sentences, without any markdown or formatting.

Entities:

[ "PERSON NAME", "ORGANIZATION", "LOCATION", "DATE", "TIME", "DURATION", "EMAIL", "PHONE NUMBER", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP CODE", "CURRENCY", "PRICE", "PRODUCT", "SERVICE", "BRAND", "EVENT", "PERCENTAGE", "AGE", "TEMPERATURE", "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "SPEED", "LANGUAGE", "NATIONALITY", "RELIGION", "JOB TITLE", "COMPANY NAME", "DEVICE NAME", "OPERATING SYSTEM", "SOFTWARE VERSION", "COLOR", "SHAPE", "MATERIAL", "MODEL NUMBER", "LICENSE PLATE", "VEHICLE MAKE", "VEHICLE MODEL", "VEHICLE TYPE", "FLIGHT NUMBER", "HOTEL NAME", "BOOKING REFERENCE", "PAYMENT METHOD", "CREDIT CARD NUMBER", "ACCOUNT NUMBER", "INSURANCE PROVIDER", "POLICY NUMBER", "BANK NAME", "TAX ID", "SOCIAL SECURITY NUMBER", "DRIVER'S LICENSE", "PASSPORT NUMBER", "WEBSITE", "URL", "IP ADDRESS", "MAC ADDRESS", "USERNAME", "PASSWORD", "FOOD ITEM", "DRINK ITEM", "CUISINE", "INGREDIENT", "DISH NAME", "MENU ITEM", "ORDER NUMBER", "PAYMENT AMOUNT", "DELIVERY TIME", "DELIVERY DATE", "APPOINTMENT DATE", "APPOINTMENT TIME", "ROOM NUMBER", "HOSPITAL NAME", "DOCTOR NAME", "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST NAME", "TEST RESULT", "INSURANCE PLAN", "CLAIM NUMBER", "POLICY HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY CONTACT", "PROJECT NAME", "TASK", "MEETING", "AGENDA", "ACTION ITEM", "DEADLINE", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT", "QUESTION", "RESPONSE" ]

Example:

Input Sentences:
[
    "ഈ കാബ്ബേജ് മത്തങ്ങ അങ്ങനെയൊക്കെ ഇള്ള സാധങ്ങള് എനിക്ക് കിട്ടുന്നുണ്ട് ഇതും ഇതിന് എൻ്റെ അടുത്തുള്ള ഒരു ചന്തയ്ണ്ട് മാർക്കെറ്റില് പോയി ഇത് വിൽക്കുകയും അത്യാവശ്യം നല്ല പൈസ കിട്ടുകയും ചെയ്യുന്നു ഇതിന് വന്നിട്ട് ഞാൻ യൂസ് ചെയ്യുന്നത് ജൈവ",
    "ഈ ഓടാൻ പോകുന്ന ആളുടെ ശരീരത്തിൽ പറഞ്ഞാൽ കൊഴുപ്പൊന്നും ഉണ്ടാകില്ല അവരെപ്പോഴും വർക്കൗട്ട് ചെയ്യുന്നത് കൊണ്ട് തന്നെ എപ്പളും ഫിറ്റായിരിക്കും",
    "ഓക്കേ അപ്പൊ ബില്ല് കിട്ടീട്ടുണ്ട് ആഹ് ഓക്കേ മാം പ്രൊഡക്ടിന് എന്തൊക്കെ ആയിരുന്നു ഡാമേജ് കാര്യങ്ങൾ ഇണ്ടായിരുന്നത്",
    "കമ്പ് നട്ട് കമ്പ് നട്ട് അങ്ങനെ രണ്ടുമൂന്ന് ചെടിയൊക്കെ ആക്കി എടുക്കും പിന്നെ എലച്ചെടികളൊക്കെ കൂടുതലായിട്ട്ണ്ടെങ്കില് അതിൻ്റെയൊക്കെ തണ്ട് ചെറുതായിട്ട് മുറിച്ച് അല്ലെങ്കില് വിത്തൊക്കെ ഇണ്ടെങ്കില് അത് തന്നെയെടുത്ത് സൂക്ഷിച്ച് വെച്ച് പാകി അങ്ങനെയാണ് ചെടി കൂടുതലായും ഞാന് പുതിയ പുതിയ ചെടികള് വളർത്തിയെടുക്കുന്നത്",
    "അറ്റത്തു നിന്ന് ഒരു എന്താ പറയുക പ്ലാസ്റ്റിക് ഉരുകുന്നപോലേന്ന് പറയാൻ പറ്റത്തില്ല ഒരു പ്ലാസ്റ്റിക്കിൻ്റെ ആ ഒരു ടെക്സ്ചർ അങ്ങ് മാറി"
]

Tagged Output:
[
    "ഈ ENTITY_FOOD_ITEM കാബ്ബേജ് END ENTITY_FOOD_ITEM മത്തങ്ങ END അങ്ങനെയൊക്കെ ഇള്ള സാധങ്ങള് എനിക്ക് കിട്ടുന്നുണ്ട് ഇതും ഇതിന് എൻ്റെ അടുത്തുള്ള ഒരു ENTITY_LOCATION ചന്തയ്ണ്ട് മാർക്കെറ്റ് END ൽ പോയി ഇത് വിൽക്കുകയും അത്യാവശ്യം ENTITY_PRICE നല്ല പൈസ END കിട്ടുകയും ചെയ്യുന്നു ഇതിന് വന്നിട്ട് ഞാൻ യൂസ് ചെയ്യുന്നത് ജൈവ",
    "ഈ ഓടാൻ പോകുന്ന ആളുടെ ശരീരത്തിൽ പറഞ്ഞാൽ കൊഴുപ്പൊന്നും ഉണ്ടാകില്ല അവരെപ്പോഴും വർക്കൗട്ട് ചെയ്യുന്നത് കൊണ്ട് തന്നെ എപ്പളും ഫിറ്റായിരിക്കും",
    "ഓക്കേ അപ്പൊ ബില്ല് കിട്ടീട്ടുണ്ട് ആഹ് ഓക്കേ മാം ENTITY_PRODUCT പ്രൊഡക്ടിന് END എന്തൊക്കെ ആയിരുന്നു ഡാമേജ് കാര്യങ്ങൾ ഇണ്ടായിരുന്നത്",
    "കമ്പ് നട്ട് കമ്പ് നട്ട് അങ്ങനെ രണ്ടുമൂന്ന് ചെടിയൊക്കെ ആക്കി എടുക്കും പിന്നെ ENTITY_INGREDIENT എലച്ചെടികളൊക്കെ END കൂടുതലായിട്ട്ണ്ടെങ്കില് അതിൻ്റെയൊക്കെ തണ്ട് ചെറുതായിട്ട് മുറിച്ച് അല്ലങ്കിൽ വിത്തൊക്കെ ഇണ്ടെങ്കില് അത് തന്നെയടുത്ത് സൂക്ഷിച്ച് വെച്ച് പാകി അങ്ങനെയാണ് ചെടി ശ്രദ്ധിച്ച് പുതിയ പുതിയ ചെടികള് വളർത്തിയെടുക്കുന്നത്",
    "അറ്റത്തു നിന്ന് ഒരു എന്താ പറയുക ENTITY_MATERIAL പ്ലാസ്റ്റിക് END ഉരുകുന്നപോലേന്ന് പറയാൻ പറ്റത്തില്ല ഒരു ENTITY_MATERIAL പ്ലാസ്റ്റിക് END ന്റെ ആ ഒരു ടെക്സ്ചർ അങ്ങ് മാറി"

]
Input Sentences:
{json.dumps(sentences, ensure_ascii=False)}
'''
    job_request = {
        "custom_id": str(idx),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini-2024-07-18",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "n": 1,
            "temperature": 0.5
        }
    }
    json.dump(job_request, job_file, ensure_ascii=False)
    job_file.write("\n")

# Create job file and process batches
with open(job_path, 'a', encoding='utf-8') as job_file:
    for batch_num in range(0, total_records, batch_size):
        batch_records = records[batch_num:batch_num + batch_size]
        sentences = [record['text'] for record in batch_records if record['path'] not in annotated_paths]
        paths = [record['path'] for record in batch_records if record['path'] not in annotated_paths]

        if not sentences:
            print(f"Skipping already annotated batch {batch_num // batch_size + 1}.")
            continue

        print(f"Processing batch {batch_num // batch_size + 1}...")
        write_job_request(sentences, batch_num // batch_size, job_file)

# Create batch request on OpenAI API
client = openai.OpenAI(api_key='YOUR_API_KEY')
input_file = client.files.create(
    file=open(job_path, "rb"),
    purpose="batch"
)
input_file_id = input_file["id"]

client.batches.create(
    input_file_id=input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "nightly eval job"}
)
print("Batch request submitted successfully.")
