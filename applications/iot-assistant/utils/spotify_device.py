import yaml
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load configuration from YAML file
def load_config():
    with open('config.yml', 'r') as file:
        return yaml.safe_load(file)

def get_spotify_client():
    """ Create a Spotify client using OAuth with settings from config. """
    config = load_config()
    spotify_config = config['spotify']
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=spotify_config['client_id'],
        client_secret=spotify_config['client_secret'],
        redirect_uri=spotify_config['redirect_uri'],
        scope=spotify_config.get('scope', 'user-read-playback-state user-modify-playback-state')
    ))

def test_playback(device_id, song_uri='spotify:track:4uLU6hMCjMI75M1A2tKUQC'):  # Default song URI for testing
    sp = get_spotify_client()
    try:
        sp.start_playback(device_id=device_id, uris=[song_uri])
        print(f"Started playback on device ID: {device_id}")
    except Exception as e:
        print(f"Failed to start playback on device ID: {device_id}, error: {e}")

def main():
    sp = get_spotify_client()
    devices = sp.devices()
    print("Available devices:", devices)
    for device in devices['devices']:
        print(f"Device Name: {device['name']} - Device ID: {device['id']}")
        # Uncomment the next line to test playback on each device automatically
        # test_playback(device['id'])

if __name__ == '__main__':
    main()

