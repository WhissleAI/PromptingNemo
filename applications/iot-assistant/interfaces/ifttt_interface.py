import requests
import json
import os
import yaml
from pathlib import Path

class IFTTTAgent:
    def __init__(self):
        # Load configuration from YAML file
        config_path = Path(__file__).parent.parent / "config.yml"
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        
        self.webhook_key = config['ifttt']['webhook_key']
        self.base_url = "https://maker.ifttt.com/trigger"
    
    def trigger_event(self, event_name, value1=None, value2=None, value3=None):
        """
        Trigger an IFTTT webhook event
        :param event_name: Name of the IFTTT event to trigger
        :param value1: Optional first value to pass to IFTTT
        :param value2: Optional second value to pass to IFTTT
        :param value3: Optional third value to pass to IFTTT
        """
        url = f"{self.base_url}/{event_name}/with/key/{self.webhook_key}"
        
        data = {}
        if value1 is not None:
            data['value1'] = value1
        if value2 is not None:
            data['value2'] = value2
        if value3 is not None:
            data['value3'] = value3
            
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return f"Successfully triggered IFTTT event: {event_name}"
        except requests.exceptions.RequestException as e:
            return f"Error triggering IFTTT event: {str(e)}"

def ifttt_agent(user_input):
    """
    Process user input and trigger appropriate IFTTT events
    :param user_input: User's command or request
    :return: Response message
    """
    agent = IFTTTAgent()
    
    # Extract intent and parameters from user input
    # This is a simple example - you might want to use more sophisticated NLP
    input_lower = user_input.lower()
    
    # Handle different IoT scenarios
    if "turn on" in input_lower and "lights" in input_lower:
        return agent.trigger_event("turn_on_lights")
    
    elif "turn off" in input_lower and "lights" in input_lower:
        return agent.trigger_event("turn_off_lights")
    
    elif "set temperature" in input_lower:
        # Extract temperature value using simple string parsing
        # You might want to use regex or better parsing methods
        try:
            temp = next(float(word) for word in input_lower.split() if word.replace('.','').isdigit())
            return agent.trigger_event("set_temperature", value1=str(temp))
        except (StopIteration, ValueError):
            return "Please specify a valid temperature value"
    
    elif "lock" in input_lower and "door" in input_lower:
        return agent.trigger_event("lock_door")
    
    elif "unlock" in input_lower and "door" in input_lower:
        return agent.trigger_event("unlock_door")
    
    elif "security" in input_lower and ("arm" in input_lower or "enable" in input_lower):
        return agent.trigger_event("arm_security")
    
    elif "security" in input_lower and ("disarm" in input_lower or "disable" in input_lower):
        return agent.trigger_event("disarm_security")
    
    return "I'm not sure what IoT action you want to perform. Please be more specific."