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

### 📹 Pretrained Models

Currently, we provide models capable of tagging different tokens while transcribing.

| Language            | Token-type             | #hrs        | #parameters     | HF-link                                                     |
|---------------------|------------------------|-------------|----------------|-------------------------------------------------------------|
| English             | NER, Emotion           | 2500 hrs    | 110M            | [Speech-Tagger-EN](https://huggingface.co/bert-base-uncased) |
| English             | 19 entities, Emotion   | 1000 hrs    | 85M             | [Cnformer-CTC](https://huggingface.co/bert-base-uncased)     |
| EURO (5 languages)  | Entities, Emotion      | CommonVoice | 85M             | [Cnformer-CTC](https://huggingface.co/bert-base-uncased)     |



## Usage

To use these pretrained models, follow the instructions provided in the source links. Typically, you'll need to install specific libraries or frameworks, such as TensorFlow, PyTorch, or the Hugging Face Transformers library.

For example, to use the BERT Base model from Hugging Face, you can do the following in Python:

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("WhissleAI/stt_en_conformer_ctc_digits")

transcriptions = asr_model.transcribe(["file.wav"])
```







### Synthetic tagged dataset creation

#### 🏷️ Tagging Real Audio Data
- **Using Pretrained Tagger**: Tag real audio data with pretrained models and standard ASR datasets. [Scripts](./scripts/data/real)
- **Using LLM for Tagging**: Utilize language models to tag synthetic data. [Scripts](./scripts/data/synthetic)

#### 🧪 Synthetic Data for ASR-NL System
- **Data Generation**: Generate synthetic datasets to train ASR and natural language systems.

#### Tagging Audio-visual data

### 🛠️ Fine-tuning with new data
- **Fine-tuning**: Enhance your ASR models with custom fine-tuning processes.
- **Evaluation**: Evaluate the performance of your fine-tuned models.

### 📤 Model Export
- **Exporting Nemo Models**: Convert your Nemo models to ONNX and Hugging Face formats for deployment.

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

## 🤝 Contributing
We welcome contributions to PromptingNemo! Please read our [contribution guidelines](CONTRIBUTING.md) to get started.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](./LICENCE) file for details.

## 🙏 Acknowledgements
Special thanks to all contributors and the open-source community for their invaluable support.
