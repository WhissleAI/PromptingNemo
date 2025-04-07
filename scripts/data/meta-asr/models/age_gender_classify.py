import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

class AgeGenderClassifier:
    def __init__(self, model_name="audeering/wav2vec2-large-robust-6-ft-age-gender", device=None):
       
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  
        
    def preprocess_audio(self, audio_path=None, signal=None, sampling_rate=16000):

        if audio_path is not None:
            signal, sr = librosa.load(audio_path, sr=sampling_rate)
        elif signal is not None:
            sr = sampling_rate
        else:
            raise ValueError("Either audio_path or signal must be provided")
            
        return signal, sr
        
    def process_audio(self, signal, sampling_rate, embeddings=False):

        inputs = self.processor(signal, sampling_rate=sampling_rate)
        input_values = inputs['input_values'][0]
        input_values = input_values.reshape(1, -1)
        input_values = torch.from_numpy(input_values).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            if embeddings:
                result = outputs[0] 
            else:
                result = torch.hstack([outputs[1], outputs[2]]) 

        return result.detach().cpu().numpy()
        
    def predict(self, audio_path=None, signal=None, sampling_rate=16000, embeddings=False):

        signal, sr = self.preprocess_audio(audio_path, signal, sampling_rate)
      
        output = self.process_audio(signal, sr, embeddings)
        
        if embeddings:
            return {"embeddings": output}
        else:
            age_pred = float(output[0][0]) * 100  
            gender_pred = output[0][1:]  
            gender_idx = np.argmax(gender_pred)
            gender_labels = ["female", "male", "other"]
            
            return {
                "age": round(age_pred, 2),
                "gender": gender_labels[gender_idx],
                "gender_probabilities": {
                    label: float(prob) 
                    for label, prob in zip(gender_labels, gender_pred)
                }
            }
            
classifier = AgeGenderClassifier()
results = classifier.predict(audio_path="speaker_segments/speaker_SPEAKER_00_segment_4.wav")
print("Results from audio file:", results)
signal, sr = librosa.load("speaker_segments/speaker_SPEAKER_00_segment_4.wav", sr=16000)
results = classifier.predict(signal=signal, sampling_rate=sr)
print("Results from signal:", results)
embeddings = classifier.predict(audio_path="speaker_segments/speaker_SPEAKER_00_segment_4.wav", 
                              embeddings=True)
print("Embeddings shape:", embeddings["embeddings"].shape)