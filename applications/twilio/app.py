from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse
import yaml
import logging
import eventlet
from flask_socketio import SocketIO, emit
import websockets
import asyncio
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Load configuration from YAML file
def load_config():
    try:
        with open("config.yml", "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        logging.info("Configuration loaded successfully.")
        return cfg
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

# Load configuration
config = load_config()

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = config['FLASK']['SECRET_KEY']

# Initialize SocketIO for WebSocket support
socketio = SocketIO(app, async_mode='eventlet')

# Twilio and ASR WebSocket configurations
TWILIO_ACCOUNT_SID = config['TWILIO']['ACCOUNT_SID']
TWILIO_AUTH_TOKEN = config['TWILIO']['AUTH_TOKEN']
TWILIO_AUTH_HEADER = config['TWILIO']['AUTH_HEADER']
ASR_WEBSOCKET_URL = config['ASR']['WEBSOCKET_URL']  # e.g., wss://api.whissle.ai/socket.io/?EIO=4&transport=websocket

@app.route("/answer", methods=['POST'])
def answer_call():
    """Respond to incoming call and start streaming audio."""
    try:
        logging.debug("Processing /answer request.")
        resp = VoiceResponse()

        # Start a <Stream> to send audio to a WebSocket server
        start = resp.start()
        start.stream(
            name="Example Audio Stream",
            url="wss://6a94-35-222-28-146.ngrok-free.app/stream_audio",  # Use wss:// for WebSocket
            track="inbound"
        )

        # Notify the caller
        resp.say("The bot is listening. Please start speaking.")

        # Keep the call alive
        resp.pause(length=60)

        return str(resp)
    except Exception as e:
        logging.error(f"Error in /answer: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/stream_audio", methods=['GET'])
def validate_stream_audio():
    """HTTP GET request handler to validate the /stream_audio endpoint."""
    logging.info("GET request received for /stream_audio. This route is for WebSocket connections only.")
    return "This endpoint is for WebSocket connections only.", 404


@socketio.on('connect', namespace='/stream_audio')
def handle_connect():
    """Handle WebSocket connection."""
    logging.info("WebSocket connection established for /stream_audio.")
    emit('connected', {'message': 'WebSocket connected successfully'})


@socketio.on('audio_in', namespace='/stream_audio')
def handle_audio(data):
    """Handle incoming audio data and forward it to the ASR WebSocket server."""
    logging.info(f"Received {len(data)} bytes of audio data.")

    async def forward_audio_to_asr(audio_data):
        """Forward audio data to the ASR WebSocket server and handle transcription."""
        try:
            async with websockets.connect(ASR_WEBSOCKET_URL) as ws:
                # Send audio data to the ASR server
                logging.info("Sending audio data to ASR WebSocket server.")
                await ws.send(audio_data)

                # Wait for the transcription response
                response = await ws.recv()
                transcription = json.loads(response).get('transcript', 'No transcription received')
                logging.info(f"Transcription from ASR server: {transcription}")

                # Emit transcription back to Twilio client
                emit('transcript', {'transcript': transcription}, namespace='/stream_audio')
        except Exception as e:
            logging.error(f"Error forwarding audio to ASR server: {e}")
            emit('transcript', {'transcript': 'Error processing audio'}, namespace='/stream_audio')

    # Use an eventlet thread to handle the async task
    eventlet.spawn(forward_audio_to_asr, data)


@socketio.on('disconnect', namespace='/stream_audio')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logging.info("WebSocket connection closed.")


if __name__ == "__main__":
    try:
        logging.info("Starting Flask-SocketIO server...")
        socketio.run(app, debug=True, port=config['SERVER']['PORT'])
    except Exception as e:
        logging.critical(f"Critical error starting the server: {e}")
