# PromptingNemo
PromptingNemo is a toolkit for fine-tuning end-to-end (E2E) automatic speech recognition (ASR) systems, integrating both audio-visual and speech-only models. 

## Speech-2-Action

Focus on this repository for Speech-2-Action models, built by innovaiton and research over pretrained audio and visual encoders.

These models process streamed audio or video input and output streamed tokens. Tokens are either:
1. What you speak (standard transcription)
2. Infered TAGS

For the sample modles we provide, TAGS infered from the context belong to multi-modal language and tonal understanding, visualQA and voice biometrics. 

We provides tools for exporting Nemo models to ONNX and Hugging Face formats.

Also checkout our applications on sample usage of Speech-2-Action models


## Setting up

### Using Docker

Use pre-built docker
``` 
docker pull WhissleAI:nemo:latest
```
Build docker from WhissleAI's Nemo branch
```
add docker commands
```

## 📹 Pretrained Models

Currently, we provide models capable of tagging different tokens while transcribing.

| Language            | Token-type             | #hrs        | #parameters     | HF-link                                                     |
|---------------------|------------------------|-------------|----------------|-------------------------------------------------------------|
| Punjabi             | Transcription, key entities, age, gender, intent, dialect           | 100 hrs    | 110M            | [Speech-Tagger-PA-KEY](https://huggingface.co/WhissleAI/stt_pa_conformer_ctc_meta) |
| Hindi             | Transcription, key entities, age, gender, intent, dialect           | 100 hrs    | 110M            | [Speech-Tagger-HI-KEY](https://huggingface.co/WhissleAI/stt_hi_conformer_ctc_meta) |
| English             | Transcription, NER, Emotion           | 2500 hrs    | 110M            | [Speech-Tagger-EN-NER](https://huggingface.co/WhissleAI/speech-tagger_en_ner_emotion) |
| English             | IOT entities and emotion   | 150 hrs    | 115M             | [Speech-Tagger-EN-IOT](https://huggingface.co/WhissleAI/speech-tagger_en_slurp_iot)     |
| EURO (5 languages)  | Transcription, Entities, Emotion      | CommonVoice | 115M             | [Speech-Tagger-EURO-NER](https://huggingface.co/bert-base-uncased)     |
| English  | Transcription, Role, Entities, Emotion, Intent      | [Speech-medical-exams](https://huggingface.co/datasets/WhissleAI/speech-simulated-medical-exams) | 115M             | [Speech-Tagger-2person-medical-exams](https://huggingface.co/WhissleAI/speech-tagger_en_2person_medical_exams)     |



## Usage

### Works for <3 min samples.
This directly fetches the model from hugging-face


```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("WhissleAI/speech-tagger_en_ner_emotion")

transcriptions = asr_model.transcribe(["file.wav"])
```

### Long recording inference

```
cd scripts/nemo
python
```

## Fine-tuning

Assuming you are in the docker. 

Adjust things in ```config.yml``` to point to correct pretrained checkpoint and data manifest

```
cd /PromptingNemo/scripts/nemo/asr/
python nemo_adapter.py
```

## Data annotation and preparation

### Annotating Using NLP tagger and classifiers

```
cd /PromptingNemo/scripts/data/audio/1person/
```
This folder has scripts to process some widely used ASR and SLU datasets

```
python process_cv.py ### common-voice transcription dataset
python process_libre.py ## Multi-lingual Librespeech telphonic conversations
python process_slurp.py ## IOT focused speech focused benchmark
```


### Annotate using LLMs

![Custom Light Red Text](https://dummyimage.com/500x30/ffcccc/000000&text=Extend+an+existing+tagged+text+dataset)


```
cd PromptingNemo
python scripts/data/audio/1person/synthetic/create_synthetic_tagged_text.py
```
Change ```extension_dataset``` to a file which has a sentence on every line which has capitalized tags along with transcription. 


![Custom Red Text](https://dummyimage.com/500x30/ff0000/ffcccc/000000&text=Create+synthetic+audio)

Provide a dataset which has audio files, to choose to clone from.

Tagged sentence data in the same language.

```
python scripts/data/audio/1person/synthetic/create_tts_manifest_xtts.py
```

In this one you have to change paths in ```self.clone_voices```  and your noise files data ```self.all_noise_files```  and also set required paths to ```config```


### Transcribe and annotate


### Annotate Role-based turn-taking conversations

```
cd /PromptingNemo/scripts/data/audio/2person/
```
This folder has scripts to process role-based turn-taking conversations.

#### Audio and Role marked transcription available
When the audio recording is available with role marked transcriptions, you can annotate them and fine-tune a model with it.

```
python 01_create_manifest_raw.py ## takes folder with transcript and audio files to create a manifest
python 02_ctm2segments.py ## takes output of nemo forced aligner and organized them to segment level
python 03_annotate_turns.py ## annotate text using LLMs
python 04_split_and_emotion.py ## split audio file using timestamps, and get speaker emotion
```

or follow ```2person.ipynb``` notebook.

#### Transcription not available


### Synthetic audio and automated annotation

 Generate synthetic datasets to train ASR and natural language systems.

#### Tagging Audio-visual data


### 📤 Model Export
- **Exporting Nemo Models**: Convert your Nemo models to ONNX and Hugging Face formats for deployment.

```
python scripts/utils/nemo2hf.py
python scripts/utils/nemo2onxx.py
```

## 🛠️ Applications
Explore various applications built with PromptingNemo.

### 🗣️ VoiceBot: Speech-to-Speech AI Engine
Run the VoiceBot application using Docker.

#### Running VoiceBot
1. Build and run the Docker container:
    ```bash
    docker ...
    ```
2. Inside the Docker container:
    ```bash
    cd PromptingNemo/applications/voicebot
    python app.py
    ```

#### 📞 Call Routing for VoiceBot
- **Description**: Implement call-routing functionality within the VoiceBot for efficient handling of calls.

### 🛠️ Data Generator: Generate Synthetic Datasets
- **Tool**: Use the data generator tool to create synthetic datasets for various ASR applications.

## 🚀 Getting Started
### Prerequisites
- 🐳 Docker
- 🐍 Python 3.x
- 📦 Necessary Python libraries (listed in `requirements.txt`)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/WhissleAI/PromptingNemo.git
    ```
2. Navigate to the project directory:
    ```bash
    cd PromptingNemo
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Related Publications

Karan, S., Shahab, J., Yeon-Jun, K., Andrej, L., Moreno, D. A., Srinivas, B., & Benjamin, S. (2023, December). 1-step Speech Understanding and Transcription Using CTC Loss. In Proceedings of the 20th International Conference on Natural Language Processing (ICON) (pp. 370-377).

Karan, S., Mahnoosh, M., Daniel, P., Ryan, P., Srinivas, C. B., Yeon-Jun, K., & Srinivas, B. (2023, December). Combining Pre trained Speech and Text Encoders for Continuous Spoken Language Processing. In Proceedings of the 20th International Conference on Natural Language Processing (ICON) (pp. 832-842).


## 🤝 Contributing
We welcome contributions to PromptingNemo! Please read our [contribution guidelines](CONTRIBUTING.md) to get started.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](./LICENCE) file for details.

## 🙏 Acknowledgements
Special thanks to all contributors and the open-source community for their invaluable support.
