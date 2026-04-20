"""Robust audio-to-BPE dataset and patched collate function."""

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment


class RobustAudioToBPEDataset(audio_to_text.AudioToBPEDataset):
    skip_audio_validation_default: bool = False
    keyphrase_oversample_factor_default: float = 0.0
    default_lang_field: str = 'lang'

    def __init__(self, manifest_filepath, *args, **kwargs):
        allowed_langs = kwargs.pop('allowed_langs', None)
        if allowed_langs is not None:
            allowed_langs = {lang.upper() for lang in allowed_langs}
        lang_field = kwargs.pop('lang_field', None)
        if not lang_field:
            lang_field = getattr(self, 'default_lang_field', 'lang')

        # Build a normalized temporary manifest that guarantees language casing
        kept_lines = 0
        processed_lines = 0
        tmp_manifest = tempfile.NamedTemporaryFile('w', delete=False, suffix='.json', encoding='utf-8')
        tmp_path = Path(tmp_manifest.name)
        progress_every = int(os.environ.get('ROBUST_DATASET_PROGRESS_EVERY', '50000'))
        skip_audio_validation = self.skip_audio_validation_default or str(manifest_filepath).endswith('.validated.json')
        with open(manifest_filepath, 'r', encoding='utf-8') as src:
            for raw_line in src:
                processed_lines += 1
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Manifest contains malformed JSON at line {processed_lines}: {raw_line[:100]}"
                    ) from exc

                audio_file = data.get('audio_filepath')
                if not audio_file:
                    raise RuntimeError("Manifest entry is missing 'audio_filepath'")

                if not skip_audio_validation:
                    audio_path = Path(audio_file)
                    if not audio_path.exists():
                        raise FileNotFoundError(f"Audio file not found for manifest entry: {audio_file}")

                lang_value = data.get(lang_field)
                if not lang_value and lang_field != 'lang':
                    lang_value = data.get('lang')
                if not lang_value:
                    raise RuntimeError(
                        f"Manifest entry missing language field '{lang_field}' (and fallback 'lang') at line {processed_lines}"
                    )

                lang_value = str(lang_value).upper()
                data['lang'] = lang_value
                if lang_field != 'lang':
                    data[lang_field] = lang_value
                if not skip_audio_validation:
                    try:
                        AudioSegment.from_file(audio_file)
                    except Exception as audio_exc:
                        raise RuntimeError(
                            f"Failed to decode audio file '{audio_file}' referenced in manifest: {audio_exc}"
                        ) from audio_exc

                tmp_manifest.write(json.dumps(data, ensure_ascii=False) + '\n')
                kept_lines += 1

                if progress_every > 0 and processed_lines % progress_every == 0:
                    logging.info(
                        "Manifest prep progress [%s]: processed=%d kept=%d",
                        manifest_filepath,
                        processed_lines,
                        kept_lines,
                    )
        tmp_manifest.close()

        if kept_lines == 0:
            raise RuntimeError(
                f"No usable manifest entries remained after filtering for language IDs. Ensure your manifests contain a '{lang_field}' field for each sample."
            )

        logging.info(
            "Finished preparing dataset manifest %s: total_rows=%d kept=%d",
            manifest_filepath,
            processed_lines,
            kept_lines,
        )

        # Initialize the parent dataset with the filtered manifest
        super().__init__(manifest_filepath=str(tmp_path), *args, **kwargs)

        # Track the temporary manifest for optional cleanup
        self._filtered_manifest_path = tmp_path
        self.lang_field = lang_field

        # Build language ids aligned with the filtered manifest collection
        self.language_ids = []
        new_collection = []
        for item in self.manifest_processor.collection:
            lang_id = getattr(item, 'lang', None)
            if not lang_id and self.lang_field and self.lang_field != 'lang':
                lang_id = getattr(item, self.lang_field, None)
            if not lang_id:
                raise RuntimeError("Manifest sample is missing language metadata after normalization")
            lang_id = lang_id.upper()
            self.language_ids.append(lang_id)
            new_collection.append(item)

        self.manifest_processor.collection = new_collection

        factor = float(self.keyphrase_oversample_factor_default)
        if factor > 0.0 and new_collection:
            weights = []
            for item in new_collection:
                text = getattr(item, 'text', '') or ''
                tokens = text.split()
                entity_count = sum(1 for tok in tokens if tok.upper().startswith('ENTITY_'))
                end_count = sum(1 for tok in tokens if tok.upper() == 'END')
                score = entity_count + end_count
                weights.append(1.0 + factor * score)
            self.sample_keyphrase_weights = np.array(weights, dtype=np.float32)
        else:
            self.sample_keyphrase_weights = np.ones(len(new_collection), dtype=np.float32)

    def __del__(self):
        # Attempt to remove the temporary manifest file when the dataset is garbage collected
        tmp_path = getattr(self, '_filtered_manifest_path', None)
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as exc:
            if index < len(self.manifest_processor.collection):
                sample = self.manifest_processor.collection[index]
                audio_path = getattr(sample, 'audio_filepath', 'unknown')
            else:
                audio_path = 'unknown'
            logging.warning(
                f"Skipping dataset sample at index {index} with audio '{audio_path}' due to error: {exc}"
            )
            return None


# Monkey-patch the original NeMo class with our robust version.
# This is necessary because the model's setup helper functions are hardcoded
# to use the base AudioToBPEDataset and ignore the `_target_` config parameter.
audio_to_text.AudioToBPEDataset = RobustAudioToBPEDataset


def patched_speech_collate_fn(batch, pad_id):
    """
    A patched version of _speech_collate_fn that correctly handles mixed
    mono/stereo audio and pads all signals to a consistent length, and filters out problematic samples.
    """
    has_weight = False
    has_sample_ids = False
    for item in batch:
        if item is None:
            continue
        tuple_len = len(item)
        if tuple_len >= 6:
            has_weight = True
            has_sample_ids = True
            break
        if tuple_len >= 5:
            candidate = item[4]
            if torch.is_tensor(candidate):
                if candidate.dtype.is_floating_point:
                    has_weight = True
                else:
                    has_sample_ids = True
            elif isinstance(candidate, (float, np.floating)):
                has_weight = True
            elif isinstance(candidate, (int, np.integer)):
                has_sample_ids = True
        break

    original_len = len(batch)
    batch = [item for item in batch if item is not None]
    if len(batch) < original_len:
        logging.warning(f"Skipped {original_len - len(batch)} samples in a batch.")

    if not batch:
        empty_audio = torch.empty(0, 0, dtype=torch.float32)
        empty_lengths = torch.tensor([], dtype=torch.long)
        empty_transcripts = torch.empty(0, 0, dtype=torch.long)
        outputs = [empty_audio, empty_lengths, empty_transcripts, empty_lengths.clone()]
        if has_weight:
            outputs.append(torch.tensor([], dtype=torch.float32))
        if has_sample_ids:
            outputs.append(torch.tensor([], dtype=torch.long))
        return tuple(outputs)

    if isinstance(batch[0], list):
        batch = [item for sublist in batch for item in sublist]

    audio_signal, audio_lengths, transcript, transcript_lengths = [], [], [], []
    weights = [] if has_weight else None
    sample_ids = [] if has_sample_ids else None

    for i, sample in enumerate(batch):
        sig = sample[0]
        # Always process as raw audio, handling multi-channel cases.
        if sig.ndim > 1:
            # Audio is multi-channel, convert to mono by averaging channels.
            sig = torch.mean(sig, dim=-1)

        audio_signal.append(sig.squeeze())
        audio_lengths.append(sample[1])
        transcript.append(sample[2])
        transcript_lengths.append(sample[3])
        if has_weight and has_sample_ids and len(sample) >= 6:
            weights.append(sample[4])
            sample_idx = sample[5]
        elif has_weight and len(sample) >= 5 and not has_sample_ids:
            weights.append(sample[4])
            sample_idx = None
        elif has_sample_ids and len(sample) >= 5 and not has_weight:
            sample_idx = sample[4]
        else:
            sample_idx = None

        if has_sample_ids and sample_idx is not None:
            if torch.is_tensor(sample_idx):
                sample_idx = int(sample_idx.item())
            else:
                sample_idx = int(sample_idx)
            sample_ids.append(sample_idx)
        if has_weight and weights is not None and len(weights) > 0:
            if torch.is_tensor(weights[-1]):
                weights[-1] = float(weights[-1].item())

    # Padding logic for 1D raw audio
    max_len = max(audio_lengths) if audio_lengths else 0
    audio_signal_padded = []
    for sig in audio_signal:
        sig_padded = torch.zeros(max_len, dtype=sig.dtype, device=sig.device)
        len_to_copy = min(sig.size(0), max_len)
        sig_padded[:len_to_copy] = sig[:len_to_copy]
        audio_signal_padded.append(sig_padded)

    audio_signal = torch.stack(audio_signal_padded) if audio_signal_padded else torch.tensor([])
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    transcript_lengths = torch.tensor(transcript_lengths, dtype=torch.long)
    transcript = nn.utils.rnn.pad_sequence(transcript, batch_first=True, padding_value=pad_id)

    outputs = [audio_signal, audio_lengths, transcript, transcript_lengths]
    if has_weight:
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        outputs.append(weights_tensor)
    if has_sample_ids:
        sample_ids_tensor = torch.tensor(sample_ids, dtype=torch.long)
        outputs.append(sample_ids_tensor)

    return tuple(outputs)


# Monkey-patch the collate function
audio_to_text._speech_collate_fn = patched_speech_collate_fn
