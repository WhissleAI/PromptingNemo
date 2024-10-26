import subprocess
from pydub import AudioSegment
from pydub.playback import play
import os

def generate_speech(text):
    url = 'https://ram-valid-walleye.ngrok-free.app/v0/conversation/TTS'
    # Properly quote the entire header and data fields to ensure correct parsing
    curl_command = [
        'curl', url,
        '-H', 'Content-Type: multipart/form-data',
        '--form-string', f'text={text}',
        '--output', 'output.wav'
    ]

    print("Running command:", " ".join(curl_command))

    # When passing to subprocess, join the command into a single string
    result = subprocess.run(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode == 0 and os.path.exists('output.wav') and os.path.getsize('output.wav') > 0:
        print("Audio generated successfully.")
        return 'output.wav'
    else:
        print("Error generating audio:", result.stderr.decode())
        return None

def play_audio_content(filepath):
    try:
        audio = AudioSegment.from_file(filepath, format="wav")
        play(audio)
    except Exception as e:
        print("Failed to play the audio:", str(e))

def text_to_speech(text):
    if text:
        filepath = generate_speech(text)
        if filepath:
            play_audio_content(filepath)

# # Example usage:
# text_to_speech("who is the president of usa")
