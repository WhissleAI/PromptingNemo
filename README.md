# PromptingNemo
E2E ASR paired with a language model fine-tuned using CTC model

## Audio-visual Models
Works for audio-visual applications

- Noise aware audio-visual ASR
- Scene aware audio-visual ASR

## Speech-only Models
For applications, which are speech only

### Tagging Real audio data

- Using pretrained tagger and standard ASR datasets [Scripts](./scripts/data/real)
  

- Using LLM to tag data [Scripts](./scripts/data/synthetic)

### Synthetic data for ASR-NL system


### ASR Systems Fine-tuning

- Fine-tuning
- Evaluation

### Exporting Nemo model to ONNX, Export to HF



## Applications

### VoiceBot: Speech-2-Speech AI Engine

```
docker ..
```
Inside the docker
```
cd PromptingNemo/applications/voicebot
python app.py
```

#### Call-Routing for Voicebot


### Data-Generator: Generate synthetic datasets


