# PromptingNemo

## Overview
PromptingNemo is a comprehensive toolkit for building and fine-tuning end-to-end (E2E) automatic speech recognition (ASR) systems, integrating both audio-visual and speech-only models. This repository offers scripts and models to facilitate noise-aware and scene-aware ASR, tagging real and synthetic audio data, and fine-tuning ASR systems. Additionally, it provides tools for exporting Nemo models to ONNX and Hugging Face formats.

## Audio-visual Models
Designed for audio-visual applications, these models leverage both audio and visual data for improved ASR performance.

### Noise-aware Audio-visual ASR
- **Data Preparation**: Prepare your data using the [Data Prep Repository](https://github.com/WhissleAI/visual_speech_recognition).
- **Model Fine-tuning**: Fine-tune your audio-visual models with noise awareness.

### Scene-aware Audio-visual ASR
- **Model Fine-tuning**: Develop ASR systems that are aware of the scene context for better performance.

## Speech-only Models
These models are optimized for applications that rely solely on speech input.

### Tagging Real Audio Data
- **Using Pretrained Tagger**: Tag real audio data with pretrained models and standard ASR datasets. [Scripts](./scripts/data/real)
- **Using LLM for Tagging**: Utilize language models to tag synthetic data. [Scripts](./scripts/data/synthetic)

### Synthetic Data for ASR-NL System
- **Data Generation**: Generate synthetic datasets to train ASR and natural language systems.

### ASR Systems Fine-tuning
- **Fine-tuning**: Enhance your ASR models with custom fine-tuning processes.
- **Evaluation**: Evaluate the performance of your fine-tuned models.

### Model Export
- **Exporting Nemo Models**: Convert your Nemo models to ONNX and Hugging Face formats for deployment.

## Applications
Explore various applications built with PromptingNemo.

### VoiceBot: Speech-to-Speech AI Engine
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

#### Call Routing for VoiceBot
- **Description**: Implement call-routing functionality within the VoiceBot for efficient handling of calls.

### Data Generator: Generate Synthetic Datasets
- **Tool**: Use the data generator tool to create synthetic datasets for various ASR applications.

## Getting Started
### Prerequisites
- Docker
- Python 3.x
- Necessary Python libraries (listed in `requirements.txt`)

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

## Contributing
We welcome contributions to PromptingNemo! Please read our [contribution guidelines](CONTRIBUTING.md) to get started.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Special thanks to all contributors and the open-source community for their invaluable support.
