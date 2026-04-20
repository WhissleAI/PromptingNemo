import openai
import os
import time
import json
from datetime import datetime
from interfaces.email_interface import send_email
from interfaces.spotify_interface import spotify_agent
from interfaces.twilio_sms_interface import sms_agent
from interfaces.weather_interface import weather_agent
from interfaces.whatsapp_interface import whatsapp_agent
from interfaces.duckduckgo_interface import duckduckgo_agent
#from interfaces.calendar_interface import calendar_agent
from interfaces.stock_interface import stock_agent
from interfaces.reminder_interface import reminder_agent
from interfaces.notes_interface import notes_agent  # Import the notes agent
from interfaces.message_interface import message_agent  # Import the message agent
from interfaces.screenshot_interface import screenshot_agent  # Import the screenshot agent
from interfaces.browser_interface import browser_agent  # Import the browser agent
from interfaces.notion_interface import notion_agent  # Import the notion agent
from text_to_speech import text_to_speech
import yaml
import requests
import speech_recognition as sr
import threading

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Assign variables from the config
openai.api_key = config['openai']['api_key']

class WakeWordDetector:
    def __init__(self, wake_word="hey noddy"):
        self.wake_word = wake_word.lower()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.phrase_time_limit = 1.5  # Limit each listen to 2 seconds

    def listen_for_wake_word(self):
        print("Listening for wake word...")
        while True:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("Ready to receive audio")
                audio = self.recognizer.listen(source, phrase_time_limit=self.phrase_time_limit)
            try:
                speech = self.recognizer.recognize_google(audio).lower()
                print(f"Detected speech: {speech}")
                if self.wake_word in speech:
                    print("Wake word detected!")
                    return True
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return False

wake_word_detector = WakeWordDetector(wake_word="hello")

# Chat agent
class Chat:
    def __init__(self, model, system="You are a helpful assistant", max_tokens=500, speech=False, temp=0.7):
        openai.api_key = config['openai']['api_key']
        self.model = model
        self.speech = speech
        self.system = system
        self.max_tokens = max_tokens
        self.temp = temp
    def __str__(self):
        name = "Chat Agent [" + self.model + "]"
        return name
    def reinsert_system_message(self, messages):
        if len(messages) == 0 or (len(messages) > 0 and messages[0].get("role") != "system"):
            messages.insert(0, {"role": "system", "content": self.system})
        return messages
    def chat(self, messages):
        messages = self.reinsert_system_message(messages)
        completion = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            messages=messages
        )
        reply_content = completion.choices[0].message.content
        return reply_content
    def stream_chat(self, messages, delay_time=0.01):
        messages = self.reinsert_system_message(messages)
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            messages=messages,
            stream=True
        )
        reply_content = ''
        chunk = ''
        for event in response:
            event_text = event['choices'][0]['delta']
            new_text = event_text.get('content', '')
            print(new_text, end='', flush=True)
            reply_content += new_text
            chunk += new_text
            # Check if the chunk ends with a sentence-ending punctuation
            if chunk and chunk[-1] in {'.', '!', '?'}:
                if self.speech == True:
                    text_to_speech(chunk)
                    chunk = ''
            time.sleep(delay_time)
        # Call the ElevenLabs API for the remaining text if any
        if self.speech == True:
            text_to_speech(chunk)
            return reply_content
        return reply_content

# Allows saving of message history to file for later retrival
def write_message_history_to_file(full_message_history, directory):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"message_history_{timestamp}.json"
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as outfile:
        json.dump(full_message_history, outfile, indent=2)

# Initialize the Nvidia API details
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "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"

def nvidia_exec(message_history):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    filtered_message_history = [msg for msg in message_history if msg['role'] != 'system']

    # Prepare the messages for the Nvidia API
    data = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": "Refine the user input and output only the JSON format. Do not include any additional explanation or text. The JSON should contain the intent and entities, ensuring the text is clean and structured."},
            *filtered_message_history  # Pass the entire message history here
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024
    }
    
    # Send the request to Nvidia's API
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        refined_output = response.json()['choices'][0]['message']['content'].strip()
        print("Refined input:", refined_output)
        
        # Now use the refined output for the next stage
        data["messages"] = [
            {"role": "system", "content": "Analyze user input, and output the name of function to fulfill a user's needs. INTENT-<type> provides meaningful intent of the user. Entity-<type> marks spans of key information. \
            The spotify_agent command can play and pause songs on the user's computer, or go to the next song. \
            If the user just says, 'pause' or 'next song' or 'volume to x' that means the spotify_agent is needed. \
            Only call the spotify_agent if the user actually wants to play music, not just talk about it. \
            The send_email command will let a user send an email. The send_sms command will let a user send an SMS message. \
            The get_weather command will let a user get the current weather for a specific location. \
            The send_whatsapp command will let a user send a message via WhatsApp. \
            The get_stock_price command will let a user get the latest stock price. \
            The notes_agent command will let a user create or search notes. \
            The notion_agent command will let a user handle any document related input. \
            The message_agent command will let a user send or read messages. \
            The screenshot_agent command will let a user take a screenshot. \
            The reminder_agent command will let a user set a reminder. \
            The browser_agent command will let a user interact with a browser to search, open websites, click elements, and more. \
            The duckduckgo_agent command will let a user perform a quick web search using DuckDuckGo. \
            If none of these commands are needed, reply only with 'duckduckgo_agent'. If it is unclear, reply with 'duckduckgo_agent'. \
            You are only allowed to output one command without any extra text. \
            The only commands you are allowed to output are: 'spotify_agent', 'send_email', 'send_sms', \
            'analyze_documents', 'get_weather', 'send_whatsapp', 'get_calendar', 'get_stock_price', 'notes_agent', 'notion_agent', 'message_agent', 'screenshot_agent', 'browser_agent', 'duckduckgo_agent', or 'chat'. Do not reply with any other output."},
            {"role": "user", "content": refined_output}  # Use the refined output here
        ]
        
        response = requests.post(API_URL, headers=headers, json=data)

        
        if response.status_code == 200:
            command = response.json()['choices'][0]['message']['content'].strip()
            print("Command:", command)

            # Call the respective agent function with the refined input
            agent_dict = { 
                "spotify_agent": spotify_agent,
                "send_email": send_email,
                "send_sms": sms_agent,
                "get_weather": weather_agent,
                "send_whatsapp": whatsapp_agent,
                "get_stock_price": stock_agent,
                "notes_agent": notes_agent,
                "notion_agent": notion_agent,
                "message_agent": message_agent,
                "screenshot_agent": screenshot_agent,
                "browser_agent": browser_agent,
                "duckduckgo_agent": duckduckgo_agent,
                "reminder_agent": reminder_agent
            }
            
            if command in agent_dict:
                if command == "spotify_agent":
                    agent_response = agent_dict[command](message_history[-1]["content"])
                else:
                    agent_response = agent_dict[command](refined_output)
            else:
                agent_response = "chat"
            
            return agent_response
        else:
            print(f"Error: {response.status_code}, {response.text}")
    else:
        print(f"Error: {response.status_code}, {response.text}")


# def gpt4_exec(message_history):
#     # Convert the full message history to JSON or a string format as needed
#     #user_input = message_history  # This is the full conversation history now

#     refined_input = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         temperature=0,
#         messages=[
#             {"role": "system", "content": "Refine the user input to make it standardized and clean. \
#                 The user input is transcribed ASR text, which may contain disfluencies. Remove these disfluencies and clean up the text. \
#                 The input contains INTENT-<type> that indicates the intent and Entity-<type> that marks spans of key information. \
#                 Ensure all key words or phrases are captured, and correct any incomplete or incorrect tags. \
#                 Organize this into a JSON format with the intent and entities, and ensure the text is clean and structured."},
#             *message_history  # Pass the entire message history here
#         ]
#     )
#     refined_input = refined_input.choices[0].message.content.strip()
#     print("Refined input:", refined_input)

#     completion = openai.ChatCompletion.create(
#         model="gpt-4",
#         temperature=0,
#         max_tokens=10,
#         messages=[
#             {"role": "system", "content": "Analyze user input, and output the name of function to fulfill a user's needs. INTENT-<type> provides meaningful intent of the user. Entity-<type> marks spans of key information. \
#             The spotify_agent command can play and pause songs on the user's computer, or go to the next song. \
#             If the user just says, 'pause' or 'next song' or 'volume to x' that means the spotify_agent is needed. \
#             Only call the spotify_agent if the user actually wants to play music, not just talk about it. \
#             The send_email command will let a user send an email. The send_sms command will let a user send an SMS message. \
#             The get_weather command will let a user get the current weather for a specific location. \
#             The send_whatsapp command will let a user send a message via WhatsApp. \
#             The get_stock_price command will let a user get the latest stock price. \
#             The notes_agent command will let a user create or search notes. \
#             The notion_agent command will let a user create or search notion entries. \
#             The message_agent command will let a user send or read messages. \
#             The screenshot_agent command will let a user take a screenshot. \
#             The browser_agent command will let a user interact with a browser to search, open pages, click elements, and more. \
#             The duckduckgo_agent command will let a user perform a quick web search using DuckDuckGo. \
#             If none of these commands are needed, reply only with 'duckduckgo_agent'. If it is unclear, reply with 'duckduckgo_agent'. \
#             You are only allowed to output one command. \
#             The only commands you are allowed to output are: 'spotify_agent', 'send_email', 'send_sms', \
#             'analyze_documents', 'get_weather', 'send_whatsapp', 'get_calendar', 'get_stock_price', 'notes_agent', 'notion_agent', 'message_agent', 'screenshot_agent', 'browser_agent', 'duckduckgo_agent', or 'chat'. Do not reply with any other output."},
#             {"role": "user", "content": refined_input}
#         ] 
#     )
    
#     command = completion.choices[0].message.content.strip()
#     command = command.replace("'", "")
#     print("Command:", command)
#     #refined_input = refinement.choices[0].message.content.strip()

#     # Call the respective agent function with the refined input
#     agent_dict = { 
#         "spotify_agent": spotify_agent,
#         "send_email": send_email,
#         "send_sms": sms_agent,
#         "get_weather": weather_agent,
#         "send_whatsapp": whatsapp_agent,
#         "get_stock_price": stock_agent,
#         "notes_agent": notes_agent,
#         "notion_agent": notion_agent,
#         "message_agent": message_agent,
#         "screenshot_agent": screenshot_agent,
#         "browser_agent": browser_agent,
#         "duckduckgo_agent": duckduckgo_agent
#     }
    
#     if command in agent_dict:
#         agent_response = agent_dict[command](refined_input)
#     else:
#         agent_response = "chat"
    
#     return agent_response


def transcribe_audio(audio_file_path):
    url = "https://related-wildcat-hugely.ngrok-free.app/transcribe-web2"
    headers = {
        'Authorization': 'Bearer random_token'
    }
    files = {
        'audio': open(audio_file_path, 'rb'),
        'language_id': (None, 'EN_IOT')
    }
    response = requests.post(url, headers=headers, files=files)
    print(response.text)
    if response.status_code == 200:
        return response.json().get('transcript', '')
    else:
        print("Error transcribing audio:", response.text)
        return None

# Global variable to control speech interruption
interrupt_speech = False

def text_to_speech_with_interrupt(text):
    global interrupt_speech
    interrupt_speech = False
    def speak():
        for chunk in text.split('.'):
            if interrupt_speech:
                break
            text_to_speech(chunk)
            time.sleep(0.5)  # Adjust sleep time as needed
    t = threading.Thread(target=speak)
    t.start()
    return t

def stop_speech():
    global interrupt_speech
    interrupt_speech = True


def main_text():
    try:
        print("Welcome to the Whissle Assistant interface!")
        text_to_speech("Welcome to the Whissle Assistant interface!")
        text_to_speech("Say 'hello' to start a conversation.")
        print("Type 'quit' to exit the chat.\n")
        speech = True
        message_history = []
        full_message_history = []
        system_message = "You are Whissle. Whissle is an AI assistant. Your name is Whissle. \
                        You can chat, send emails, get weather information, interact with Spotify, \
                        send WhatsApp messages, manage calendar events, get stock prices, and manage notes. \
                        You can also interact with Notion to create and search notes or other entries. \
                        Above all, you enjoy having interesting, intellectually stimulating \
                        conversations."
        max_history = 100  # Adjust this value to limit the number of messages considered

        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        # Initialize the wake word detector
        #wake_word_detector = WakeWordDetector(wake_word="hello vortex")

        while True:
            # Wait for the wake word before listening for user input
            if not wake_word_detector.listen_for_wake_word():
                continue

            with microphone as source:
                print("Listening for your input...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=30, phrase_time_limit=5)
                
            # Stop any ongoing speech
            stop_speech()

            # try:
            with open("input.wav", "wb") as f:
                f.write(audio.get_wav_data())

            user_input = transcribe_audio("input.wav")
            print("You:", user_input)
            if not user_input:
                print("Sorry, I did not understand that. Please try again.")
                continue

            print(f"You: {user_input}")
            # except sr.UnknownValueError:
            #     print("Sorry, I did not understand that. Please try again.")
            #     continue
            # except sr.RequestError:
            #     print("Sorry, my speech recognition service is down. Please try again later.")
            #     continue

            if user_input.lower() == 'quit':
                write_message_history_to_file(full_message_history, "./message_logs")
                break
            else:
                message_history.append({"role": "user", "content": user_input})
                full_message_history.append({"role": "user", "content": user_input})
                # Reduces messages when max history exceeded
                if len(message_history) > max_history:
                    message_history = [message_history[0]] + message_history[-(max_history - 1):]  # Keep the system message and remove the second message
                # Check user input, if executive is needed, call executive on user input and return result.
                
                print("Message history:", message_history)
                agent_response = nvidia_exec(message_history)
                if agent_response == False:
                    print("\nWhissle: ", end='', flush=True)
                    gpt4_chat = Chat("gpt-4", system=system_message, speech=speech)
                    response = gpt4_chat.stream_chat(message_history)
                    message_history.append({"role": "assistant", "content": response})
                    full_message_history.append({"role": "assistant", "content": response})
                    print(f"\n")
                else:
                    if isinstance(agent_response, list):  # Handling the case when the agent returns a list of responses
                        for i, response in enumerate(agent_response):
                            message_history.append(response)
                            full_message_history.append(response)
                            # Print only the most recent answer
                            if i == len(agent_response) - 1:
                                print(response["content"])
                                if speech:
                                    speech_thread = text_to_speech_with_interrupt(response["content"])
                    else:  # Handling the case when the agent returns a single response (string)
                        message_history.append({"role": "assistant", "content": agent_response})
                        full_message_history.append({"role": "assistant", "content": agent_response})
                        print(agent_response)
                        if speech:
                            speech_thread = text_to_speech_with_interrupt(agent_response)

    except KeyboardInterrupt:
        print("\nDetected KeyboardInterrupt. Saving message history and exiting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}. Saving message history and exiting.")
    finally:
        write_message_history_to_file(full_message_history, "./message_logs")
        print("Message history saved.")

if __name__ == "__main__":
    main_text()
