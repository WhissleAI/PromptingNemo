import re
import os
from typing import Dict, Callable

# Define functions for each action
def set_device_setting(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    setting_type = entities.get("ENTITY_SETTING_TYPE")
    setting_value = entities.get("ENTITY_SETTING_VALUE")
    location = entities.get("ENTITY_LOCATION")
    measurement = entities.get("ENTITY_MEASUREMENT")
    command = f"kasa set {device} --{setting_type.replace(' ', '-')} {setting_value or measurement} --location {location}"
    #os.system(command)
    print(command)

def turn_off_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"kasa set {device} --state off --location {location}"
    #os.system(command)
    print(command)

def turn_on_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"kasa set {device} --state on --location {location}"
    #os.system(command)
    print(command)

def play_music(entities: Dict[str, str]):
    content = entities.get("ENTITY_CONTENT")
    device = entities.get("ENTITY_DEVICE")
    command = f"music_system play {content} --device {device}"
    #os.system(command)
    print(command)

def send_message(entities: Dict[str, str]):
    recipient = entities.get("ENTITY_RECIPIENT")
    message = entities.get("ENTITY_CONTENT")
    command = f"twilio api:core:messages:create --from YOUR_TWILIO_NUMBER --to {recipient} --body '{message}'"
    #os.system(command)
    print(command)

def set_reminder(entities: Dict[str, str]):
    event = entities.get("ENTITY_EVENT")
    date = entities.get("ENTITY_DATE")
    time = entities.get("ENTITY_TIME")
    content = entities.get("ENTITY_CONTENT")
    command = f"gcalcli add --title '{event}' --when '{date} {time}' --description '{content}'"
    #os.system(command)
    print(command)

def create_event(entities: Dict[str, str]):
    event = entities.get("ENTITY_EVENT")
    date = entities.get("ENTITY_DATE")
    time = entities.get("ENTITY_TIME")
    command = f"gcalcli add --title '{event}' --when '{date} {time}'"
    #os.system(command)
    print(command)

def call_recipient(entities: Dict[str, str]):
    recipient = entities.get("ENTITY_RECIPIENT")
    platform = entities.get("ENTITY_PLATFORM")
    command = f"call_control call --recipient {recipient} --platform {platform}"
    #os.system(command)
    print(command)

def activate_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"security_system activate --device {device} --location {location}"
    #os.system(command)
    print(command)

def deactivate_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"security_system deactivate --device {device} --location {location}"
    #os.system(command)
    print(command)

def pause_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"media_control pause --device {device} --location {location}"
    #os.system(command)
    print(command)

def skip_content(entities: Dict[str, str]):
    content = entities.get("ENTITY_CONTENT")
    device = entities.get("ENTITY_DEVICE")
    command = f"media_control skip --device {device} --content {content}"
    #os.system(command)
    print(command)

def stop_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"appliance_control stop --device {device} --location {location}"
    #os.system(command)
    print(command)

def lock_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"kasa set {device} --state lock --location {location}"
    #os.system(command)
    print(command)

def unlock_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    command = f"kasa set {device} --state unlock"
    #os.system(command)
    print(command)

def open_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"window_control open --device {device} --location {location}"
    #os.system(command)
    print(command)

def close_device(entities: Dict[str, str]):
    device = entities.get("ENTITY_DEVICE")
    location = entities.get("ENTITY_LOCATION")
    command = f"blinds_control close --device {device} --location {location}"
    #os.system(command)
    print(command)

# Define a mapping from intents to functions
intent_to_function: Dict[str, Callable[[Dict[str, str]], None]] = {
    "INTENT_SETTING_CHANGE": set_device_setting,
    "INTENT_DEVICE_CONTROL": turn_off_device,
    "INTENT_DEVICE_CONTROL": turn_on_device,
    "INTENT_MUSIC_CONTROL": play_music,
    "INTENT_MESSAGE_SEND": send_message,
    "INTENT_REMINDER_SET": set_reminder,
    "INTENT_EVENT_CREATE": create_event,
    "INTENT_CALL": call_recipient,
    "INTENT_DEVICE_CONTROL": activate_device,
    "INTENT_DEVICE_CONTROL": deactivate_device,
    "INTENT_DEVICE_CONTROL": pause_device,
    "INTENT_DEVICE_CONTROL": skip_content,
    "INTENT_DEVICE_CONTROL": stop_device,
    "INTENT_DEVICE_CONTROL": lock_device,
    "INTENT_DEVICE_CONTROL": unlock_device,
    "INTENT_DEVICE_CONTROL": open_device,
    "INTENT_DEVICE_CONTROL": close_device
}

# Function to extract entities from a tagged sentence
def extract_entities(tagged_sentence: str) -> Dict[str, str]:
    entities = {}
    matches = re.findall(r'ENTITY_(\w+)\s(.*?)\sEND', tagged_sentence)
    for match in matches:
        entity_type, entity_value = match
        entities[f"ENTITY_{entity_type}"] = entity_value.strip('"')
    return entities

# Example usage
tagged_sentences = [
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Set END the ENTITY_SETTING_TYPE color temperature END of the ENTITY_DEVICE lights END in the ENTITY_LOCATION living room END to ENTITY_SETTING_VALUE warm white END. INTENT_SETTING_CHANGE",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Increase END the ENTITY_SETTING_TYPE brightness END of the ENTITY_DEVICE TV END in the ENTITY_LOCATION bedroom END. INTENT_SETTING_CHANGE",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Decrease END the ENTITY_SETTING_TYPE speed END of the ENTITY_DEVICE fan END in the ENTITY_LOCATION kitchen END. INTENT_SETTING_CHANGE",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Turn off END the ENTITY_DEVICE music system END in the ENTITY_LOCATION garage END. INTENT_DEVICE_CONTROL",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Mute END the ENTITY_DEVICE TV END in the ENTITY_LOCATION living room END. INTENT_DEVICE_CONTROL",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Play END the ENTITY_CONTENT song 'Imagine' END on the ENTITY_DEVICE stereo END. INTENT_MUSIC_CONTROL",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Send END a ENTITY_CONTENT message END to ENTITY_RECIPIENT John END saying 'Happy Birthday'. INTENT_MESSAGE_SEND",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Set END a ENTITY_EVENT reminder END for ENTITY_DATE tomorrow END at ENTITY_TIME 5 PM END to buy groceries. INTENT_REMINDER_SET",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Create END an ENTITY_EVENT event END named 'Dinner with friends' on ENTITY_DATE this Saturday END at ENTITY_TIME 7 PM END. INTENT_EVENT_CREATE",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Change END the ENTITY_SETTING_TYPE color END of the ENTITY_DEVICE lights END in the ENTITY_LOCATION bathroom END to ENTITY_SETTING_VALUE blue END. INTENT_SETTING_CHANGE",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Call END ENTITY_RECIPIENT Mom END on ENTITY_PLATFORM Skype END. INTENT_CALL",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Increase END the ENTITY_SETTING_TYPE volume END of the ENTITY_DEVICE speakers END in the ENTITY_LOCATION living room END. INTENT_SETTING_CHANGE",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Pause END the ENTITY_DEVICE washing machine END in the ENTITY_LOCATION laundry room END. INTENT_DEVICE_CONTROL",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Decrease END the ENTITY_SETTING_TYPE temperature END of the ENTITY_DEVICE thermostat END in the ENTITY_LOCATION bedroom END. INTENT_SETTING_CHANGE",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Activate END the ENTITY_DEVICE security system END in the ENTITY_LOCATION house END. INTENT_DEVICE_CONTROL",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Unmute END the ENTITY_DEVICE TV END in the ENTITY_LOCATION living room END. INTENT_DEVICE_CONTROL",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Skip END the ENTITY_CONTENT current song END on the ENTITY_DEVICE music player END. INTENT_MUSIC_CONTROL",
    "DOMAIN-SMARTHOME-ASSISTANT ENTITY_ACTION Decrease END the ENTITY_SETTING_TYPE brightness END of the ENTITY_DEVICE TV END in the ENTITY_LOCATION bedroom END. INTENT_SETTING_CHANGE"
]

for tagged_sentence in tagged_sentences:
    intent_match = re.search(r'INTENT_(\w+)', tagged_sentence)
    if intent_match:
        intent = f"INTENT_{intent_match.group(1)}"
        entities = extract_entities(tagged_sentence)
        if intent in intent_to_function:
            intent_to_function[intent](entities)
        else:
            print(f"No function mapped for intent: {intent}")
