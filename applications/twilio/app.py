from flask import Flask, request, render_template, Response, copy_current_request_context
from flask import session

import yaml
from flask import Flask, request, render_template, Response, session
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
import os
import time
import requests
import uuid
import subprocess

import openai

# Load configuration from YAML file
def load_config():
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

config = load_config()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config['FLASK']['SECRET_KEY']

# Disable warnings from urllib3
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Set SSL cert file from config
os.environ['SSL_CERT_FILE'] = config['SSL_CERT']['FILE']

# Twilio credentials from config
twilio_config = config['TWILIO']
twillio_account_sid = twilio_config['ACCOUNT_SID']
twillio_auth_token = twilio_config['AUTH_TOKEN']
twillio_auth_header = twilio_config['AUTH_HEADER']

# Initialize Twilio client
client = Client(twillio_account_sid, twillio_auth_token)

# print("OS PATH", os.environ['PATH'])
# print("OS ENV", os.environ)
# http_proxy = os.getenv('HTTP_PROXY')
# print('HTTP_PROXY:', http_proxy)

# # Get the value of the HTTPS_PROXY environment variable
# https_proxy = os.getenv('HTTPS_PROXY')
# print('HTTPS_PROXY:', https_proxy)


def download_file(url, headers, file_path):
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return "Download successful"
    else:
        return f"Download failed with status code {response.status_code}"


@app.route("/answer", methods=['GET', 'POST'])
def answer_call():
    """Respond to incoming calls with a simple text to speech message."""
    
    # Initialize or reset the conversation history for each new call
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Append system welcome message to the conversation history
    session['conversation_history'].append({"role": "system", "content": "Welcome to the Vortex. How can I help you?"})

    # Generate the voice response
    resp = VoiceResponse()
    resp.say("Welcome to the immersive Vortex. How can I help you?")
    resp.redirect('/record')

    # Save the session changes
    session.modified = True

    return str(resp)

@app.route("/record", methods=['GET', 'POST'])
def record_speech():
    """Handle the response from the user and save the recording."""
    response = VoiceResponse()
    
    # Start recording with optimized settings
    response.record(
        speech_timeout=1,  # Handles gaps in speech more effectively
        action='/process_speech',
        recording_channels='mono',
        recording_sample_rate='16000',
        trim="trim-silence"  # Auto-trim silent parts
    )

    # Provide immediate feedback
    response.say("Thank you for taking time talking to our dmeo. Have a nice day.")

    # Hang up or redirect as needed
    response.hangup()
    return str(response)

def get_transcription(audiofile):

    # Create a dictionary with the audio file to send in the request
    
    transcribe_url = "http://localhost:5000/transcribe_twilio"
    
    files = {'audio': open(audiofile, 'rb')}

    # Send an HTTP POST request to the /transcribe endpoint
    json_data = {'model_name': 'ner_emotion_commonvoice', 'language_id': 'EN'}
    response = requests.post(transcribe_url, files=files, data=json_data)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the transcription from the response
        data = response.json()
        transcription = data['transcript']
        token_timestamps = data['token_timestamps']
    else:
        print('Error:', response.status_code)

    return transcription, token_timestamps

def get_tensorrt_llm_response(text):
    

    # Create a dictionary with the audio file to send in the request
    
    transcribe_url = "http://localhost:5000/llm_response_tensorrt"
    
    # Send an HTTP POST request to the /transcribe endpoint
    json_data = {'content': text}
    response = requests.post(transcribe_url, data=json_data)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the transcription from the response
        data = response.json()
        response = data['response']
    else:
        print('Error:', response.status_code)

    return response
 
    
def get_chatgpt_response(text):
    openai.api_key = config['OPENAI']['API_KEY']

    # Prepare instructions and system messages correctly
    instructions = "Your name is Vortex. Using the provided text, recognize and consider named entities marked with 'NER_TYPE' labels to maintain context relevance. Additionally, adapt your response tone to align with the emotional state indicated by 'EMOTION_TYPE' tags to ensure an empathetic and appropriate engagement with the user."
    
    # Initialize or update the conversation history in session
    if 'conversation_history' not in session or not session['conversation_history']:
        # Start with a system message if the conversation history is empty
        session['conversation_history'] = [{"role": "system", "content": instructions}]

    # Ensure user input is added as a dictionary with 'role' and 'content'
    session['conversation_history'].append({"role": "user", "content": text})
    
    # Generate the response using the updated conversation history
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=session['conversation_history'],
            temperature=0.1,
            max_tokens=150
        )
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return "Error processing your request."

    # Extract text from the response and update session history
    chat_response = response['choices'][0]['message']['content']
    print("LLM -->", chat_response)
    session['conversation_history'].append({"role": "assistant", "content": chat_response})
    
    # Mark session as modified to save changes
    session.modified = True
    
    return chat_response

def clean_tags(input_text):
    input_text = input_text.split()
    new_sent = []
    for word in input_text:
        if "NER_" not in word and "END" not in word and "EMOTION_" not in word:
            new_sent.append(word)

    return " ".join(new_sent)

@app.route("/process_speech", methods=['POST'])
def process_speech():
    """Process the recorded speech by downloading it as an MP3 file and generate a response using ChatGPT."""

    # Twilio credentials
    #account_sid = "AC036b65392d3ed4c975b62dfdd94dd6d8"
    #auth_token = "67860a42a80d84e2cd2aa4e36443d616"
    client = Client(twillio_account_sid, twillio_auth_token)

    # Get the recording SID
    recording_sid = request.values.get("RecordingSid", None)
    if not recording_sid:
        return "Recording SID not provided", 400

    # Construct URL for the Twilio API request
    url = f"https://api.twilio.com/2010-04-01/Accounts/{twillio_account_sid}/Recordings/{recording_sid}.wav"
    headers = {'Authorization': 'Basic QUMwMzZiNjUzOTJkM2VkNGM5NzViNjJkZmRkOTRkZDZkODo2Nzg2MGE0MmE4MGQ4NGUyY2QyYWE0ZTM2NDQzZDYxNg=='}

    # Save the recording
    # Generate a unique filename using UUID
    unique_filename = f"recording_{recording_sid}_{uuid.uuid4()}.wav"
    file_path = os.path.join('/workspace/advanced-speech-LLM-demo/twilio/user_audio', unique_filename)
    #file_path = os.path.join('/workspace/advanced-speech-LLM-demo/twilio/user_audio', 'recording.wav')
    
    f = open(file_path, 'wb')

    # Polling to check if the recording is available
    max_attempts = 30
    wait_time = 0.01
    for attempt in range(max_attempts):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            f.write(response.content)
            f.close()
            break
        time.sleep(wait_time)
    else:
        return f"Failed to download the recording after {max_attempts} attempts.", 500


        

    # Process transcription and response
    transcription, token_timestamps = get_transcription(file_path)
    print("Transcription:", transcription)
    chat_response = get_tensorrt_llm_response(transcription)
    print("Chat Response", chat_response)
    
    # Create Twilio voice response and use Gather to allow interruptions
    resp = VoiceResponse()
    gather = Gather(input='speech', timeout=2, action='/handle_input', method='POST')
    gather.say(chat_response)  # Say the ChatGPT response within Gather
    resp.append(gather)

    # If the user doesn't say anything, redirect back to continue the process
    resp.redirect('/record')

    return str(resp)

@app.route("/handle_input", methods=['POST'])
def handle_input():
    """Handle user input after interrupting the response."""
    resp = VoiceResponse()
    if 'SpeechResult' in request.values:
        # Get the user's spoken input
        speech_input = request.values['SpeechResult'].strip()

        print("back-channel-transcription:", speech_input)
        # Use your existing function to get a response from ChatGPT
        # Assuming your get_chatgpt_response function is designed to handle this
        chat_response = get_tensorrt_llm_response(speech_input)
        #chat_response = get_chatgpt_response(speech_input)

        # Say the response generated by ChatGPT
        resp.say(chat_response)
    else:
        resp.say("Sorry, I did not catch that.")

    resp.redirect('/record')  # Redirect to continue interaction
    return str(resp)

def generate_transcription(conversation_history):
    """A generator function that yields transcriptions as they are available."""
    idx = 0
    while True:
        while idx >= len(conversation_history):
            time.sleep(1)  # Sleep to wait for more data
        if idx < len(conversation_history):
            yield conversation_history[idx]
            idx += 1

@app.route('/stream')
def stream():
    """Streams transcription data as server-sent events."""
    # Capture the current conversation history at the time of the request
    conversation_history = session.get('conversation_history', [])

    def event_stream(conversation_history):
        for transcript in generate_transcription(conversation_history):
            yield f"data: {transcript['content']}\n\n"

    return Response(event_stream(conversation_history), mimetype="text/event-stream")


@app.route('/')
def show_transcriptions():
    """Renders the HTML page for showing transcriptions."""
    return render_template('transcriptions.html')


if __name__ == "__main__":
    app.run(debug=True, port=config['SERVER']['PORT'])
