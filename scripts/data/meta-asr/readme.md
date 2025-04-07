Meta-ASR: Audio Processing Pipeline
===================================

Overview
--------

Meta-ASR is a comprehensive audio processing pipeline designed to handle various speech datasets, perform transcription, classify intents, and process multilingual speech data. The repository integrates multiple models and scripts to streamline the audio processing workflow.

Repository Structure
--------------------

```
meta-asr/
│── models/                  # Contains models for various tasks
│   │── age_gender_classify.py    # Wav2Vec2-based age and gender classification
│   │── nemo_inverse_normalize.py # Converts cardinal numbers to normal form
│   │── ner_taggers.py            # Named Entity Recognition for speech data
│   │── whissle_trans.py          # Whisper model-based transcription
│── process-data-en/          # Scripts for English speech dataset processing
│   │── process_avspeech-data.py    # Processes AVSpeech dataset
│   │── process_common_voice.py    # Processes Common Voice dataset
│   │── process_opensr.py          # Processes OpenSR dataset
│   │── process_yt_data.py         # Processes YouTube dataset
|── annote-entent/
|   │── german.py    # to annote german data
|   │── italian.py    # to annote italian data
|   │── port.py    # to annote portuguse data
|   │── spanish.py    # to annote spanish data
│   │── french.py    # to annote french dat
│── process-data-mls/         # Scripts for MultiLingual Speech (MLS) processing
│   │── mls_spanish.py         # Processes Spanish speech data
│   │── process_mls_french.py  # Processes French speech data
│   │── process_mls_german.py  # Processes German speech data
│   │── process_mls_italian.py # Processes Italian speech data
│   │── process_mls_port.py    # Processes Portuguese speech data
│── entent_classification.py   # Classifies intents and entities in speech data
│── transcribe_using_gemini.py # Transcribes audio using Gemini model
│── parquet_data_process.py    # Handles audio dataset processing in Parquet format
│── requirements.txt           # Dependencies for running the pipeline
│── readme.md                  # Documentation

```

Features
--------

-   **Audio Transcription**: Uses Whisper and Gemini models for transcription.
-   **Dataset Processing**: Supports AVSpeech, Common Voice, OpenSR, YouTube, and MLS datasets.
-   **Intent Classification**: Recognizes entities such as `PERSON_NAME`, `ORGANIZATION`, and intents like `INTENT_INFORM`.
-   **Age and Gender Prediction**: Implements Wav2Vec2-based classification.
-   **Data Normalization**: Converts numeric values to their spoken form using NeMo Inverse Normalization.
-   **Multilingual Support**: Processes speech data in Spanish, French, German, Italian, and Portuguese.

Processed Data Format
---------------------

The pipeline outputs JSONL files structured as follows:

```
{
  "audio_filepath": "/path/to/audio.wav",
  "text": "to the voice let me give you an example the other day i had a new student come to me AGE_30_45 GER_MALE EMOTION_NEU INTENT_EXPLAIN",
  "duration": 4.95
}

```

Installation
------------

1.  Clone the repository:

    ```
    git clone https://github.com/your-repo/meta-asr.git
    cd meta-asr

    ```

2.  Install dependencies:

    ```
    pip install -r requirements.txt

    ```

Usage
-----

### Transcribing Audio

```
python transcribe_using_gemini.py --audio /path/to/audio.wav

```

### Processing Datasets

```
python process-data-en/process_avspeech-data.py

```

Contributors
------------

-   **Himanshu Gangwar**

License
-------

This project is licensed under the MIT License.