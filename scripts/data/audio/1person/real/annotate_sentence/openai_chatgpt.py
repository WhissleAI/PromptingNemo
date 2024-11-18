import json
import re
import openai
import traceback


def annotate_sentences_openai(sentences):
    print("\nStarting sentence annotation")
    lang_code = "English"
    
    prompt = f'''
    Given a list of sentences in English, annotate each sentence individually with the appropriate entity tags from the provided list. The sentences may relate to various actions such as managing tasks, controlling devices, sending notifications, scheduling events, updating information, making purchases, or offering assistance.

    **Instructions:**

    - Annotate each sentence separately with entity tags based on the entities in the provided list.
    - Use the format `ENTITY_<type>` to start each entity and `END` to close it.
    - For each sentence, add a relevant intent label at the end in the format `INTENT_<type>`, where `<type>` represents the sentence's main action or goal.
    - Only use the entity types provided in the list below.
    - Do not add any additional text, explanations, or comments.
    - Ensure the output is a JSON array containing only the annotated sentences in the same order as the input sentences, without any markdown or formatting.

    **Entities**:

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
        
        "AGE", "GENDER", "NATIONALITY", "RELIGION", "MARITAL_STATUS", "OCCUPATION", "EDUCATION_LEVEL", "DEGREE", 
        "SKILL", "EXPERIENCE", "YEARS_OF_EXPERIENCE", "CERTIFICATION", 
        
        "MEASUREMENT", "DISTANCE", "WEIGHT", "HEIGHT", "VOLUME", "TEMPERATURE", "SPEED", "CAPACITY", "DIMENSION", "AREA", 
        "SHAPE", "COLOR", "MATERIAL", "TEXTURE", "PATTERN", "STYLE", 
        
        "WEATHER_CONDITION", "TEMPERATURE_SETTING", "HUMIDITY_LEVEL", "WIND_SPEED", "RAIN_INTENSITY", "AIR_QUALITY", 
        "POLLUTION_LEVEL", "UV_INDEX", 
        
        "QUESTION_TYPE", "REQUEST_TYPE", "SUGGESTION_TYPE", "ALERT_TYPE", "REMINDER_TYPE", "STATUS", "ACTION", "COMMAND"
    ]

    **Example**:

    Input Sentences:
    [
        "Can you set up a meeting with John at 3 PM tomorrow?",
        "Play some jazz music in the living room.",
        "Schedule a delivery for October 15th at 10 AM.",
        "Please book a table at the Italian restaurant for two at 7 PM.",
        "Remind me to take my medication at 9 AM."
    ]

    Expected Output:
    [
        "Can you ENTITY_ACTION set up END a ENTITY_MEETING meeting END with ENTITY_PERSON_NAME John END at ENTITY_TIME 3 PM END on ENTITY_DATE tomorrow END? INTENT_SCHEDULE_MEETING",
        
        "ENTITY_ACTION Play END some ENTITY_CATEGORY jazz music END in the ENTITY_LOCATION living room END. INTENT_MEDIA_CONTROL",
        
        "ENTITY_ACTION Schedule END a ENTITY_DELIVERY delivery END for ENTITY_DATE October 15th END at ENTITY_TIME 10 AM END. INTENT_SCHEDULE_DELIVERY",
        
        "Please ENTITY_ACTION book END a ENTITY_TABLE table END at the ENTITY_CUISINE Italian restaurant END for ENTITY_PARTY_SIZE two END at ENTITY_TIME 7 PM END. INTENT_BOOK_RESERVATION",
        
        "ENTITY_ACTION Remind END me to ENTITY_ACTION take END my ENTITY_MEDICATION medication END at ENTITY_TIME 9 AM END. INTENT_SET_REMINDER"
    ]

    **Sentences to Annotate:**
    {json.dumps(sentences, ensure_ascii=False)}
    '''
    
    openai.api_key = "openai_api_key"
    
    try:
        print("Making OpenAI API call...")
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.5
        )
        
        assistant_reply = response.choices[0].message.content
        print("Received API response")
        print("Raw response:", assistant_reply)
        
        try:
            annotated_sentences = json.loads(assistant_reply)
            if isinstance(annotated_sentences, list):
                print(f"Successfully parsed {len(annotated_sentences)} annotated sentences")
                return annotated_sentences
            else:
                print("Warning: API response was not a list after JSON parsing")
                return [assistant_reply] * len(sentences)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print("Falling back to raw reply")
            return [assistant_reply] * len(sentences)
            
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return sentences  # Return original sentences as fallback