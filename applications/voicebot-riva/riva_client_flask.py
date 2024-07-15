from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
import riva.client
import os
import traceback

app = Flask(__name__)

# Initialize Riva client authentication and ASR service
auth = riva.client.Auth(uri='localhost:50051')
riva_asr = riva.client.ASRService(auth)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_audio_content/<filename>', methods=['GET'])
def get_audio_content(filename):
    # Assuming the audio files are stored in a directory named 'audio_files'
    audio_path = os.path.join(app.root_path, 'demo_audio', filename)
    return send_file(audio_path, as_attachment=True)

@app.route('/demo_audio/<path:filename>')
def serve_audio(filename):
    try:
        return send_from_directory('demo_audio', filename)
    except Exception as e:
        traceback.print_exc()  # Print the traceback for debugging
        return str(e), 500  # Return the error message and status code

@app.route('/get_files')
def get_files():
    folder_path = 'demo_audio'  # Replace this with your folder path
    files = os.listdir(folder_path)
    return jsonify(files=files)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400

    audio_file = request.files['audio']
    content = audio_file.read()

    # Set up recognition configuration
    config = riva.client.RecognitionConfig()
    config.language_code = "en-US"
    config.max_alternatives = 1
    config.enable_automatic_punctuation = False
    config.audio_channel_count = 1
    #config.model_name = "asr_offline_conformer_ctc_pipeline"

    # Perform offline recognition
    response = riva_asr.offline_recognize(content, config)
    asr_best_transcript = response.results[0].alternatives[0].transcript

    return jsonify({'transcript': asr_best_transcript})

if __name__ == '__main__':
    app.run(debug=True, host='')
