from twilio.rest import Client
import requests

# Twilio credentials
account_sid = "AC036b65392d3ed4c975b62dfdd94dd6d8"
auth_token = "67860a42a80d84e2cd2aa4e36443d616"

# Initialize the client
client = Client(account_sid, auth_token)

# SID of the recording you want to fetch
recording_sid = 'RE498dceffa759f5207eb2c64e50bb3b33'

# Fetch the recording
recording = client.recordings(recording_sid).fetch()

# Print recording details or download the recording
print(recording.call_sid)  # Prints the Call SID associated with the recording
print(recording.duration)  # Prints the duration of the recording in seconds

# Download the recording file (MP3, WAV, etc.)
url = f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.wav'
response = requests.get(url, auth=(account_sid, auth_token))

# Ensure the request was successful
if response.status_code == 200:
    with open('recording.wav', 'wb') as f:
        f.write(response.content)  # Write the binary content to the file
    print("Recording downloaded successfully.")
else:
    print(f"Failed to download the recording. Status code: {response.status_code}")