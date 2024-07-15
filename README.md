# PromptingNemo
E2E ASR paired with a language model fine-tuned using CTC model

### Tagging Real audio data

- Using pretrained tagger
- Using LLM to tag data

### Synthetic data for ASR-NL system

This script generates diverse natural language commands for a smart home system, annotates them in tagged text using OpenAI's GPT-4, and saves the dataset to a file. Each command involves changing a setting like color temperature, brightness, speed, state, volume, playing a song, sending a message, setting a reminder, or creating an event.

#### Prerequistes
- Python 3.6 or later
- OpenAI API Key

#### 3 simple steps

- Use prompts and generate data using GPT
```
python scripts/create_synthetic_tagged_text.py
```

- Create synthetic audio and manifest file

Two options: use googleTTS then set
```
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/google-tts-key.json"
```
or use coqui TTS

```
python scripts/sysnthetic_audio_manifest.py {gtts,xtts,piper}
```

<clean_text_file>: Path to your clean text file.
<tagged_text_file>: Path to your tagged text file.
<audio_output_path>: Directory where you want to save the audio files.
<manifest_file>: Path to the output manifest file.

verify the files, now it's time to train a speech model.


### Fine-tuning ASR Systems






### Exporting Nemo model to Model_shelf (onnx models)



## Applications

### VoiceBot: Speech-2-Speech AI Engine


