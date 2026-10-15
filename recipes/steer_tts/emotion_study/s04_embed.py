"""Learned representations per clip (GPU).

1. audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim -> arousal, dominance, valence
   (continuous, ~0..1; trained on MSP-Podcast; licence CC BY-NC-SA 4.0 -> analysis/annotation use).
2. microsoft/wavlm-base-plus -> mean-pooled hidden state of every layer (13 x 768), for
   layer-wise probing (which layers carry emotion vs. sentence vs. speaker) and UMAP.

out/avd.parquet          clip, arousal, dominance, valence
out/wavlm_layers.npy     float16 [N, 13, 768] in clips.parquet order
"""
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model, WavLMModel

from common import WAV, OUT

AVD = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
WAVLM = "microsoft/wavlm-base-plus"


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))


class EmotionModel(nn.Module):
    """Backbone via the stock Wav2Vec2Model; regression head weights loaded from the checkpoint by hand
    (the model card's PreTrainedModel subclass no longer loads under transformers v5)."""

    def __init__(self, name):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(name)
        self.classifier = RegressionHead(self.wav2vec2.config)
        sd = load_checkpoint(name)
        head = {k[len("classifier."):]: v for k, v in sd.items() if k.startswith("classifier.")}
        self.classifier.load_state_dict(head)

    def forward(self, x):
        h = self.wav2vec2(x)[0].mean(1)
        return self.classifier(h)


def load_checkpoint(name):
    from huggingface_hub import hf_hub_download
    try:
        from safetensors.torch import load_file
        return load_file(hf_hub_download(name, "model.safetensors"))
    except Exception:
        return torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu")


def main():
    dev = "cuda"
    clips = pd.read_parquet(OUT / "clips.parquet")
    fe_avd = AutoFeatureExtractor.from_pretrained(AVD)
    m_avd = EmotionModel(AVD).to(dev).eval()
    fe_wl = AutoFeatureExtractor.from_pretrained(WAVLM)
    m_wl = WavLMModel.from_pretrained(WAVLM).to(dev).eval()

    rows, emb = [], np.zeros((len(clips), 13, 768), np.float16)
    for i, clip in enumerate(tqdm(clips["clip"])):
        y, _ = sf.read(WAV / f"{clip}.wav", dtype="float32")
        with torch.no_grad():
            x = torch.tensor(fe_avd(y, sampling_rate=16000).input_values[0])[None].to(dev)
            a, d, v = m_avd(x)[0].tolist()
            x = torch.tensor(fe_wl(y, sampling_rate=16000).input_values[0])[None].to(dev)
            hs = m_wl(x, output_hidden_states=True).hidden_states
            emb[i] = torch.stack([h[0].mean(0) for h in hs]).cpu().numpy()
        rows.append(dict(clip=clip, arousal=a, dominance=d, valence=v))

    avd = pd.DataFrame(rows)
    avd.to_parquet(OUT / "avd.parquet", index=False)
    np.save(OUT / "wavlm_layers.npy", emb)
    print(avd.merge(clips[["clip", "emo"]]).groupby("emo")[["arousal", "dominance", "valence"]].mean().round(3))


if __name__ == "__main__":
    main()
