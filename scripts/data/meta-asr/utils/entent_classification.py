import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def remove_existing_tags(text):
    """
    Remove legacy (NER_) and current (ENTITY_) entity tags while preserving special tags
    like GENDER_*, EMOTION_*, AGE_*, and SPEAKER_CHANGE.
    """

    preserved_tags = {
        r'GENDER_[A-Z]+': lambda m: m.group(0),
        r'EMOTION_[A-Z]+': lambda m: m.group(0),
        r'AGE_[0-9_]+': lambda m: m.group(0),
        r'SPEAKER_CHANGE': lambda m: m.group(0),
    }
    preserved_text = text
    for pattern, replacement in preserved_tags.items():
        preserved_text = re.sub(pattern, replacement, preserved_text)

    cleaned_text = re.sub(r'(?:NER|ENTITY)_\w+\s?', '', preserved_text)

    cleaned_text = re.sub(r'\s?END', '', cleaned_text)

    return cleaned_text

def fix_end_tags(text):
    """
    Ensure that every ENTITY_ annotation is followed by a space and 'END'.
    """
    if not isinstance(text, str):
        return text

    text = re.sub(r'(\w)END', r'\1 END', text)
    text = re.sub(r'(ENTITY_[A-Z0-9_]+)(\S)', r'\1 \2', text)
    pattern = r'(ENTITY_[A-Z0-9_]+\s+[^\s.,?!]+)(?!\s+END)'
    text = re.sub(pattern, r'\1 END', text)
    return text

def annotate_sentences(sentences):
    prompt = f'''
You are given a list of sentences that may contain existing tags like GENDER_FEMALE, EMOTION_NEU, AGE_45_60, and SPEAKER_CHANGE.

Your task is to:
1. Remove any legacy entity tags such as NER_PERSON, NER_NORP, etc.
2. Preserve these existing tags exactly as they appear.
3. **ONLY annotate entities from the following list.** Do NOT invent new entity types or annotate entities that are not explicitly listed.
4. Insert entity tags in the format ENTITY_<TYPE> before each entity from the list and append " END" (with a space) after it.
5. Focus on identifying and tagging entities that are specifically mentioned in the provided list.
6. Classify the **intent** of each sentence and add an **INTENT_<INTENT_TYPE>** tag at the end of the sentence.
7. Return the output as a JSON array of annotated sentences.

IMPORTANT:
- Do NOT modify or add tags for gender, emotion, age, or speaker changes.
- Remove any legacy tags (e.g., NER_PERSON) and annotate entities using **only** the ENTITY_<TYPE> format and **only from the provided list.**
- **Do NOT annotate general categories like "symptoms," "emotions," "devices," or "services" unless they directly correspond to a specific entity type in the provided list.** For example, "facial deformity" should not be annotated as ENTITY_SYMPTOM if "SYMPTOM" is not intended to be a general category for annotation from the list.  Focus on the specific entity types listed.
- Every entity annotation must be in the form "ENTITY_<TYPE> entity_text END" (notice the space before END).
- Intent annotation must be in the form "INTENT_<INTENT_TYPE>" and placed at the end of the sentence.

Entities to annotate (ONLY USE THESE ENTITY TYPES):
[
    "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE", "CURRENCY", "PRICE",
    "DATE", "TIME", "DURATION", "APPOINTMENT_DATE", "APPOINTMENT_TIME", "DEADLINE", "DELIVERY_DATE", "DELIVERY_TIME",
    "EVENT", "MEETING", "TASK", "PROJECT_NAME", "ACTION_ITEM", "PRIORITY", "FEEDBACK", "REVIEW", "RATING", "COMPLAINT",
    "QUESTION", "RESPONSE", "NOTIFICATION_TYPE", "AGENDA", "REMINDER", "NOTE", "RECORD", "ANNOUNCEMENT", "UPDATE",
    "SCHEDULE", "BOOKING_REFERENCE", "APPOINTMENT_NUMBER", "ORDER_NUMBER", "INVOICE_NUMBER", "PAYMENT_METHOD",
    "PAYMENT_AMOUNT", "BANK_NAME", "ACCOUNT_NUMBER", "CREDIT_CARD_NUMBER", "TAX_ID", "SOCIAL_SECURITY_NUMBER",
    "DRIVER'S_LICENSE", "PASSPORT_NUMBER", "INSURANCE_PROVIDER", "POLICY_NUMBER", "INSURANCE_PLAN", "CLAIM_NUMBER",
    "POLICY_HOLDER", "BENEFICIARY", "RELATIONSHIP", "EMERGENCY_CONTACT", "PROJECT_PHASE", "VERSION", "DEVELOPMENT_STAGE",
    "DEVICE_NAME", "OPERATING_SYSTEM", "SOFTWARE_VERSION", "BRAND", "MODEL_NUMBER", "LICENSE_PLATE", "VEHICLE_MAKE",
    "VEHICLE_MODEL", "VEHICLE_TYPE", "FLIGHT_NUMBER", "HOTEL_NAME", "ROOM_NUMBER", "TRANSACTION_ID", "TICKET_NUMBER",
    "SEAT_NUMBER", "GATE", "TERMINAL", "TRANSACTION_TYPE", "PAYMENT_STATUS", "PAYMENT_REFERENCE", "INVOICE_STATUS",
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "DOSAGE", "ALLERGY", "PRESCRIPTION", "TEST_NAME", "TEST_RESULT", "MEDICAL_RECORD",
    "HEALTH_STATUS", "HEALTH_METRIC", "VITAL_SIGN", "DOCTOR_NAME", "HOSPITAL_NAME", "DEPARTMENT", "WARD", "CLINIC_NAME",
    "WEBSITE", "URL", "IP_ADDRESS", "MAC_ADDRESS", "USERNAME", "PASSWORD", "LANGUAGE", "CODE_SNIPPET", "DATABASE_NAME",
    "API_KEY", "WEB_TOKEN", "URL_PARAMETER", "SERVER_NAME", "ENDPOINT", "DOMAIN",
    "PRODUCT", "SERVICE", "CATEGORY", "BRAND", "ORDER_STATUS", "DELIVERY_METHOD", "RETURN_STATUS", "WARRANTY_PERIOD",
    "CANCELLATION_REASON", "REFUND_AMOUNT", "EXCHANGE_ITEM", "GIFT_OPTION", "GIFT_MESSAGE",
    "FOOD_ITEM", "DRINK_ITEM", "CUISINE", "MENU_ITEM", "ORDER_NUMBER", "DELIVERY_ESTIMATE", "RECIPE", "INGREDIENT",
    "DISH_NAME", "PORTION_SIZE", "COOKING_TIME", "PREPARATION_METHOD",
    "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE",
    "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION",
    "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA",
    "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE",
    "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY",
    "POLLUTION_LEVEL", "UV_INDEX",
    "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND"
]

Example:
Input: "I have 15 apples and my friend david gave me 20 more. AGE_45_60 GER_MALE EMOTION_HAP SPEAKER_CHANGE"
Output: "I have ENTITY_NUMBER 15 END apples and my friend ENTITY_PERSON_NAME david END gave me ENTITY_NUMBER 20 END more. AGE_45_60 GER_MALE EMOTION_HAP SPEAKER_CHANGE INTENT_INFORM"

Input: "Sophia won 1st prize in the science competition. AGE_15_20 GEN_FEMALE EMOTION_HAP SPEAKER_CHANGE"
Output: "ENTITY_PERSON_NAME Sophia END won ENTITY_NUMBER 1st END prize in the science competition. AGE_15_20 GEN_FEMALE EMOTION_HAP SPEAKER_CHANGE INTENT_ANNOUNCE"

Input: "I think my appointment is on 10th July. AGE_40_50 GEN_FEMALE EMOTION_NEU SPEAKER_CHANGE"
Output: "I think my appointment is on ENTITY_DATE 10th July END. AGE_40_50 GEN_FEMALE EMOTION_NEU SPEAKER_CHANGE INTENT_REMEMBER"

**Example of what NOT to do:**
Input: "Because of facial deformity, she lives a life of fear and shame. AGE_45_60 GER_OTHER EMOTION_FEAR INTENT_INFORM"
Output: "Because of facial deformity, she lives a life of fear and shame. AGE_45_60 GER_OTHER EMOTION_FEAR INTENT_INFORM"  (No ENTITY_SYMPTOM annotation, as 'SYMPTOM' as a general category isn't the focus if facial deformity isn't a specific entity you are looking for, and SYMPTOM type might not be intended for broad categories like this unless defined more specifically in your desired entity types)

Input: "A device cannot see services that are in different scopes. AGE_45_60 GER_OTHER EMOTION_NEUTRAL INTENT_INFORM"
Output: "A device cannot see services that are in different scopes. AGE_45_60 GER_OTHER EMOTION_NEUTRAL INTENT_INFORM" (No ENTITY_DEVICE or ENTITY_SERVICE annotations if 'DEVICE' and 'SERVICE' are meant to be more specific types from your list and not general categories. If 'DEVICE_NAME' is in your list, and you meant specific device names, then "device" here might be too generic to tag as ENTITY_DEVICE_NAME unless it's clarified in your use case)


Sentences to Annotate:
{json.dumps(sentences, ensure_ascii=False)}
'''

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        assistant_reply = response.text.strip()

        if assistant_reply.startswith("```"):
            lines = assistant_reply.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            assistant_reply = "\n".join(lines).strip()

        try:
            annotated_sentences = json.loads(assistant_reply)
            if isinstance(annotated_sentences, list):
                return [fix_end_tags(sentence) if isinstance(sentence, str) else sentence for sentence in annotated_sentences]
            else:
                print("Error: API did not return a list")
                return sentences
        except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}")
                print("Raw response:", assistant_reply)
                return sentences
    except Exception as e:
        print(f"Error in annotation: {e}")
        return sentences

def process_jsonl_file(input_path, output_path, batch_size=10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]

    total_records = len(records)
    print(f"Total records to process: {total_records}")

    for batch_num in range(0, total_records, batch_size):
        batch_records = records[batch_num:batch_num + batch_size]
        print(f"\nProcessing batch {batch_num//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")

        original_texts = []
        for record in batch_records:
            text_content = record.get('text')
            if isinstance(text_content, dict) and 'sentence' in text_content:
                original_texts.append(remove_existing_tags(text_content['sentence']))
            elif isinstance(text_content, dict) and 'text' in text_content:
                original_texts.append(remove_existing_tags(text_content['text']))
            elif isinstance(text_content, str):
                original_texts.append(remove_existing_tags(text_content))
            else:
                original_texts.append("")

        annotated_texts = annotate_sentences(original_texts)
        with open(output_path, 'a', encoding='utf-8') as f:
            for record, annotated_text in zip(batch_records, annotated_texts):
                record['text'] = annotated_text
                json_str = json.dumps(record, ensure_ascii=False)
                f.write(json_str + '\n')

        print(f"Processed and saved batch {batch_num//batch_size + 1}")

    print(f"\nProcessing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    input_jsonl_path = "/external4/datasets/jsonl_data/ps/rem_ps.jsonl"
    output_jsonl_path = "/hydra2-prev/home/compute/workspace_himanshu/Processed_Data/gemini_metadata/english/ps_rem1.jsonl"

    process_jsonl_file(input_jsonl_path, output_jsonl_path) 
    