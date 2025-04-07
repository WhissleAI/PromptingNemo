import torch
import torch.nn as nn
import logging, math, librosa, random
from pathlib import Path
import librosa, re
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import soundfile as sf
from fastapi import UploadFile
import asyncio
import wave
import io
import webrtcvad
from pydub import AudioSegment
import yaml


CONSTANT = 1e-5

def normalize_batch(x, seq_len, normalize_type):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            if x[i, :, : seq_len[i]].shape[1] == 1:
                raise ValueError(
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                    "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
                    "feature (ex. at least `hop_length` for Mel Spectrograms)."
                )
            x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
        return (
            (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2),
            x_mean,
            x_std,
        )
    else:
        return x, x_mean, x_std

class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """
    def __init__(
        self,
        sample_rate=16000,
        n_window_size=400, 
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=512,
        preemph=0.97,
        nfilt=80,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=0.00001,
        pad_to=0,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()

        if stft_conv or stft_exact_pad:
            logging.warning(
                "Using torch_stft is deprecated and has been removed. The values have been forcibly set to False "
                "for FilterbankFeatures and AudioToMelSpectrogramPreprocessor. Please set exact_pad to True "
                "as needed."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        logging.info(f"PADDING: {pad_to}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None

        if exact_pad:
            logging.info("STFT using exact pad")
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)
        self.stft = lambda x: torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if exact_pad else True,
            window=self.window.to(dtype=torch.float),
            return_complex=True,
        )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type
        logging.debug(f"sr: {sample_rate}")
        logging.debug(f"n_fft: {self.n_fft}")
        logging.debug(f"win_length: {self.win_length}")
        logging.debug(f"hop_length: {self.hop_length}")
        logging.debug(f"n_mels: {nfilt}")
        logging.debug(f"fmin: {lowfreq}")
        logging.debug(f"fmax: {highfreq}")
        logging.debug(f"using grads: {use_grads}")
        logging.debug(f"nb_augmentation_prob: {nb_augmentation_prob}")

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length) + 1
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x, seq_len, linear_spec=False):
        seq_len = self.get_seq_len(seq_len)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect"
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        # do preemphasis
        if self.preemph is not None:
            x = x.unsqueeze(0)  # Assuming x is a 1D tensor
            x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.squeeze(0)  # Assuming x needs to be reverted back to 1D after the operation

        # disable autocast to get full range of stft values
        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x, seq_len

# Function to load SentencePiece model and vocabulary
def load_spm_model(spm_model_path):
    print("spm_model_path: ", spm_model_path)
    s = spm.SentencePieceProcessor(model_file=spm_model_path)
    vocab = [s.id_to_piece(i) for i in range(s.get_piece_size())]
    vocab_size = s.get_piece_size()
    return s, vocab, vocab_size

def get_word_boundary_indices(token_ids):
    time_stamp = 0
    non_blank_tokens = []
    for token in token_ids[0]:
        if token != 1024:
            non_blank_tokens.append((time_stamp, token))
        time_stamp += 1
    print(non_blank_tokens)

# Function to get token ids and construct the sentence
def construct_sentence(logits, spm_model,time_stride=80):
    probabilities = logits[0][0]
    token_ids = torch.argmax(torch.tensor(probabilities), dim=1).numpy()
    token_ids = np.expand_dims(token_ids, 0)
    token_ids = torch.from_numpy(token_ids)
    token_ids_array = token_ids.numpy().tolist()
    vocabs = [spm_model.id_to_piece(id) for id in range(spm_model.get_piece_size())]
    vocabs.append("BLANK")
    
    token_timestamps = []
    time_stamp = 0
    for token in token_ids_array[0]:
        if token != 1024:
            token_timestamps.append((vocabs[token],time_stamp))
        time_stamp += time_stride

    #remove leading 1024 aka blank token
    token_ids_array = token_ids_array[0][token_ids_array[0].index(next((x for x in token_ids_array[0] if x != 1024), len(token_ids_array[0]))):]
    #apply CTC decoding rules: 1. remove repitions 2. remove blanks
    cleaned_token_ids = [x for i, x in enumerate(token_ids_array) if i == 0 or x != token_ids_array[i - 1] and x != 1024]
    #print("cleaned_token_ids: ", cleaned_token_ids)
    #print("spm model", spm_model)
    sentence = spm_model.decode_ids(cleaned_token_ids)
    

    return sentence, token_timestamps

def parse_config(file_path):
    config_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split('=')
            keys = key.split('.')
            current_dict = config_dict
            for k in keys[:-1]:
                current_dict = current_dict.setdefault(k, {})
            current_dict[keys[-1]] = value.strip()
    return config_dict

def create_ort_session(model_name, model_shelf):
    model_shelf = Path(model_shelf)
    name = Path(model_name)
    model_path = model_shelf / name
    
    config = parse_config(model_path / 'magic.txt')
    
    model_onnx = config['encoder']['onnx']
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(config['onnx']['intra_op_num_threads'])
    model_onnx = str(model_path / model_onnx)
    ort_session = ort.InferenceSession(model_onnx, options)

    model_tokenizer = config['tokenizer']['model']
    model_tokenizer = str(model_path / model_tokenizer)
    spm_model, vocab, vocab_size = load_spm_model(model_tokenizer)

    # Initialize the FilterbankFeatures instance
    sample_rate = int(config['sample_rate'])
    filterbank_featurizer = FilterbankFeatures(sample_rate=sample_rate)  # You can adjust parameters here
    
    return ort_session, spm_model, filterbank_featurizer

def infer_audio_file(featurizer, ort_session, spm_model, audio_file):

    # Get the names of input and output layers
    input_names = [input.name for input in ort_session.get_inputs()]
    output_names = [output.name for output in ort_session.get_outputs()]
    
    # Load the audio file using librosa
    waveform, sample_rate = librosa.load(audio_file, sr=16000, mono=True)
    duration = librosa.get_duration(y=waveform, sr=sample_rate)

    # Assuming seq_len is the length of the waveform
    seq_len = torch.tensor([waveform.shape[0]], dtype=torch.float)
    print("Seq Len", waveform.shape)
    
    # Convert the waveform to features
    features, features_length = featurizer.forward(torch.tensor(waveform), seq_len)
    print("Features", features.shape)
    
    input_data = {
        input_names[0]: features.cpu().numpy(),  # Convert torch tensor to numpy array
        input_names[1]: features_length.cpu().numpy()  # Convert torch tensor to numpy array
    }

    # Assuming ort_session, output_names, and input_data are already defined

    # Run inference
    logits = ort_session.run([output_names[0]], input_data)
    print("Logits", logits[0][0].shape)
    # Construct and print the sentence
    sentence, token_timestamps = construct_sentence(logits, spm_model)
    
    return sentence, token_timestamps, duration

def extract_entities(input_string, token_timestamps, tag="NER"):
    
    token_timestamps = token_timestamps.split(';')
    print("token_timestamps 1 : ", token_timestamps)
    token_timestamps = [tuple(entry.split(',')) for entry in token_timestamps if entry.startswith(('NER', 'END', 'EMOTION', 'POS', 'LANGUAGE'))]

    print("token_timestamps 2: ", token_timestamps)

    ner_entities = []
    current_entity = None
    
    while token_timestamps:
        token, time = token_timestamps.pop(0)
        print("TOKEN", "TIME")
        print(token,time)
        if token.startswith('NER_'):
            
            current_entity = token
            start_time = int(time)
        elif token == 'END' and current_entity != None:
            ner_entities.append((current_entity, (start_time, int(time))))
            current_entity = None
    print("ner_entities: ", ner_entities)
    entities = {}
    # Regex pattern to identify entities and their phrases

    if tag == "NER":
        pattern = r'(NER_\w+) (.*?) END'
    else:
        pattern = r'(POS_\w+) (.*?) END'
    matches = re.findall(pattern, input_string)

    print("Tag: ", tag)
    print(matches)
    
    for match in matches:
        entity, phrase = match
        _, time = ner_entities.pop(0)
        if entity not in entities:
            entities[entity] = [{'phrase': phrase, 'indices': [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), input_string)], "time_boundary" : time}]
        else:
            entities[entity].append({'phrase': phrase, 'indices': [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), input_string)], "time_boundary" : time})

    print("Entities:", entities)
    
    html_table = "<table border='1'><tr><th>Entity_Type</th><th>Value</th></tr>"

    for entity_type, values in entities.items():
        for value in values:
            html_table += f"<tr><td>{entity_type}</td><td>{value}</td></tr>"

    html_table += "</table>"

    return html_table

def clean_string_and_extract_emotion(input_string):
    # Define the patterns to remove
    patterns_to_remove = r'\b(?:NER_[A-Z]+|EMOTION_[A-Z]+|POS_[A-Z]+|END)\b'

    # Remove words matching the patterns
    clean_string = re.sub(patterns_to_remove, '', input_string)

    # Remove extra spaces caused by removal of words
    clean_string = re.sub(r'\s+', ' ', clean_string).strip()

    # Extract emotion type
    emotion_type_match = re.search(r'\bEMOTION_([A-Z]+)\b', input_string)
    emotion_type = emotion_type_match.group(1) if emotion_type_match else None

    if emotion_type:
        emotion_type = emotion_type.lower()

    return clean_string.strip(), emotion_type


def extract_entities_s2s(input_string, token_timestamps, tag="NER"):
    #print("token_timestamps : ", token_timestamps)
    token_timestamps = [entry for entry in token_timestamps if entry[0].startswith(('NER', 'END', 'EMOTION', 'POS', 'LANGUAGE'))]

    print("All Tags timestamps:", token_timestamps)

    ner_entities = []
    current_entity = None
    
    while token_timestamps:
        token, time = token_timestamps.pop(0)
        print(token,time)
        if token.startswith('NER_'):
            
            current_entity = token
            start_time = int(time)
        elif token == 'END' and current_entity != None:
            ner_entities.append((current_entity, (start_time, int(time))))
            current_entity = None
    print("Entities boundries:", ner_entities)
    entities = {}
    # Regex pattern to identify entities and their phrases

    if tag == "NER":
        pattern = r'(NER_\w+) (.*?) END'
    else:
        pattern = r'(POS_\w+) (.*?) END'
    matches = re.findall(pattern, input_string)

    #print("Tag: ", tag)
    #print(matches)
    
    for match in matches:
        entity, phrase = match
        _, time = ner_entities.pop(0)
        if entity not in entities:
            entities[entity] = [{'phrase': phrase, 'indices': [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), input_string)], "time_boundary" : time}]
        else:
            entities[entity].append({'phrase': phrase, 'indices': [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), input_string)], "time_boundary" : time})
    print("Entities:", entities)
    
    html_table = "<table border='1'><tr><th>Entity_Type</th><th>Value</th></tr>"

    for entity_type, values in entities.items():
        for value in values:
            html_table += f"<tr><td>{entity_type}</td><td>{value}</td></tr>"

    html_table += "</table>"

    return html_table


def extract_entities_web(input_string, token_timestamps, tag="NER"):
    #print("token_timestamps : ", token_timestamps)
    token_timestamps = [entry for entry in token_timestamps if entry[0].startswith(('NER', 'END', 'EMOTION', 'POS', 'LANGUAGE'))]

    print("All Tags timestamps:", token_timestamps)

    ner_entities = []
    current_entity = None
    
    while token_timestamps:
        token, time = token_timestamps.pop(0)
        print(token,time)
        if token.startswith('NER_'):
            
            current_entity = token
            start_time = int(time)
        elif token == 'END' and current_entity != None:
            ner_entities.append((current_entity, (start_time, int(time))))
            current_entity = None
    print("Entities boundries:", ner_entities)
    entities = {}
    # Regex pattern to identify entities and their phrases

    if tag == "NER":
        pattern = r'(NER_\w+) (.*?) END'
    else:
        pattern = r'(POS_\w+) (.*?) END'
    matches = re.findall(pattern, input_string)

    #print("Tag: ", tag)
    #print(matches)
    
    for match in matches:
        entity, phrase = match
        _, time = ner_entities.pop(0)
        if entity not in entities:
            entities[entity] = [{'phrase': phrase, 'indices': [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), input_string)], "time_boundary" : time}]
        else:
            entities[entity].append({'phrase': phrase, 'indices': [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), input_string)], "time_boundary" : time})
    print("Entities:", entities)

    return entities


async def preprocess_audio(audio: UploadFile):
    cmd = ['ffmpeg', '-hide_banner -loglevel error', '-i', "-", '-ac', '1', '-ar','16000', '-f', 'wav', '-']
    cmd=" ".join(cmd)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    audio_file,error = await proc.communicate(audio)

    if error: raise HTTPException(status_code=400, detail="error parsing uploaded file")

    return audio_file

def load_ambernet_model_config(onnx_model_path, config_path):
    ort_session = ort.InferenceSession(onnx_model_path)
    filterbank_featurizer = FilterbankFeatures(sample_rate=16000)
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
    labels = data['train_ds'].get('labels', None)
    if labels is not None:
        labels = list(labels)

    return ort_session, filterbank_featurizer, labels

def infer_ambernet_onnx(ort_session, filterbank_featurizer, labels, audio_file):
    input_names = [input.name for input in ort_session.get_inputs()]
    output_names = [output.name for output in ort_session.get_outputs()]

    waveform, sample_rate = librosa.load(audio_file, sr=16000, mono=True)
    seq_len = torch.tensor([waveform.shape[0]], dtype=torch.float)

    features, features_length = filterbank_featurizer.forward(torch.tensor(waveform), seq_len)

    input_data = {
        input_names[0]: features.cpu().numpy(),
        input_names[1]: features_length.cpu().numpy()
    }

    logits = ort_session.run([output_names[0]], input_data)

    probabilities = logits[0]

    label_id = torch.argmax(torch.tensor(probabilities), dim=1)

    return labels[label_id]

def read_wave(path_or_fileobject):
    """Reads an audio file and converts it to 16kHz, mono, 16-bit PCM."""
    audio = AudioSegment.from_file(path_or_fileobject)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit PCM
    raw_data = np.array(audio.get_array_of_samples(), dtype=np.int16).tobytes()
    return raw_data, audio.frame_rate

def write_wave_to_memory(audio_data, sample_rate):
    """Writes audio data to an in-memory file (BytesIO) as a WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    buffer.seek(0)  # Reset buffer position for reading
    return buffer

async def extract_first_n_seconds_of_audio(audio, duration_seconds, vad_mode=3):
    """Extracts the first `duration_seconds` seconds of voiced speech and returns it as a BytesIO object."""
    vad = webrtcvad.Vad(vad_mode)
    frame_duration = 30  # ms
    sample_rate = 16000
    frame_size = int(sample_rate * frame_duration / 1000) * 2  # 16-bit PCM

    audio = await preprocess_audio(audio)
    audio_data, _ = read_wave(io.BytesIO(audio))
    voiced_audio = bytearray()
    voiced_duration = 0  
    max_voiced_duration = duration_seconds * 1000  

    for i in range(0, len(audio_data), frame_size):
        frame = audio_data[i:i + frame_size]
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate):
            voiced_audio.extend(frame)
            voiced_duration += frame_duration
            if voiced_duration >= max_voiced_duration:
                break

    return write_wave_to_memory(voiced_audio, sample_rate)

