import openai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import os
import psutil
import yaml
import subprocess
import requests


# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up API key from the config
openai.api_key = config['openai']['api_key']

NVAPI_URL = config['nvidia']['api_url']
NVAPI_KEY = config['nvidia']['api_key']

def is_spotify_running():
    for process in psutil.process_iter():
        try:
            if 'spotify' in process.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def open_spotify():
    # if not is_spotify_running():
        spotify_path = config['system']['spotify_path']
        subprocess.call(['open', spotify_path])
        time.sleep(1)
    # else:
    #     pass

def get_spotify_client():
    """ Create a Spotify client using OAuth with settings from config. """
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=config['spotify']['client_id'],
        client_secret=config['spotify']['client_secret'],
        redirect_uri=config['spotify']['redirect_uri'],
        scope="user-modify-playback-state user-read-playback-state"
    ))

def play_song(song_query):
    sp = get_spotify_client()
    results = sp.search(q=song_query, limit=3, type='track')
    if not results['tracks']['items']:
        print("No song found")
        return
    song_uri = sorted(results['tracks']['items'], key=lambda x: x['popularity'], reverse=True)[0]['uri']
    device_id = config['spotify'].get('device_id', None)
    if device_id:
        try:
            sp.start_playback(device_id=device_id, uris=[song_uri])
        except Exception as e:
            print(f"No device found or error in playback: {e}")

def spotify_agent(prompt):
    # Combine the main instruction and the cleaning instruction into a single system message
    system_message = """
    You control a music app based on a user request. 
    Please clean the input prompt by removing unnecessary tags and extra information. Last word tells about speaker emotion.
    Focus on extracting the key action or command related to music playback, such as playing a song, genre, artist, or controlling playback (e.g., pause, resume, next) and emotional state.
    Ignore any irrelevant information and rephrase the reply concisely.
    """

    headers = {
        "Authorization": f"Bearer {NVAPI_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "nv-mistralai/mistral-nemo-12b-instruct",
        "messages": [
            {
                "role": "system", 
                "content": system_message
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 100  # Adjust as needed
    }

    response = requests.post(NVAPI_URL, headers=headers, json=data)
    
    try:
        response_data = response.json()
        print("Response JSON:", response_data)  # Debugging: print the entire response
        reply_content = response_data['choices'][0]['message']['content'].strip()
        print("Reply:", reply_content)
    except KeyError as e:
        print(f"KeyError: {e} - The response structure was not as expected.")
        print("Response content:", response.text)
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Response content:", response.text)
        return

    open_spotify()
    
    if "|" in reply_content:
        action, value = reply_content.split('|')
        control_playback(action.strip(), int(value.strip()))
    elif reply_content in ["pause", "next", "resume"]:
        control_playback(reply_content)
    else:
        play_song(reply_content)
        return f"Playing {reply_content}"

# # Example usage:
# spotify_agent("play something by Coldplay")
