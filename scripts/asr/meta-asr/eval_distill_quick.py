"""Quick WER eval on N samples from a distilled student checkpoint.

Usage:
    python scripts/asr/meta-asr/eval_distill_quick.py \
        --checkpoint /path/to/student.nemo \
        --manifest /path/to/valid.json \
        --n 50
"""

import argparse
import json
import logging
import os
import re
import sys

import torch
import soundfile as sf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

TAG_PATTERN = re.compile(
    r'\s*(AGE_\S+|GENDER_\S+|EMOTION_\S+|INTENT_\S+|ENTITY_\S+|END|LANG_\S+|KEYWORD_\S+)\s*'
)


def strip_tags(text: str) -> str:
    """Remove all meta-tags from text, keeping only the spoken content."""
    cleaned = TAG_PATTERN.sub(' ', text)
    return ' '.join(cleaned.split()).strip()


def word_error_rate(ref: str, hyp: str):
    """Compute WER between reference and hypothesis strings."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    r_len = len(ref_words)
    h_len = len(hyp_words)

    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[r_len][h_len], r_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to student .nemo checkpoint')
    parser.add_argument('--manifest', required=True, help='Validation manifest JSONL')
    parser.add_argument('--n', type=int, default=50, help='Number of samples to evaluate')
    parser.add_argument('--with-tags', action='store_true', help='Include tags in WER (debug)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model: extract weights + vocab from .nemo, build lightweight inference
    import tarfile, yaml, tempfile, glob
    logging.info("Loading student from %s", args.checkpoint)

    tmpdir = tempfile.mkdtemp()
    with tarfile.open(args.checkpoint, 'r:') as tar:
        tar.extractall(tmpdir)

    # Find the checkpoint file
    ckpt_files = glob.glob(os.path.join(tmpdir, '**', '*.ckpt'), recursive=True)
    if not ckpt_files:
        ckpt_files = glob.glob(os.path.join(tmpdir, '**', 'model_weights.ckpt'), recursive=True)
    if not ckpt_files:
        # Try .pt files
        ckpt_files = glob.glob(os.path.join(tmpdir, '**', '*.pt'), recursive=True)
    logging.info("Found checkpoint files: %s", ckpt_files)

    state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=False)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Load config
    cfg_path = os.path.join(tmpdir, 'model_config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Extract vocabulary from tokenizer files in archive
    vocab_files = glob.glob(os.path.join(tmpdir, '**', '*vocab*'), recursive=True)
    logging.info("Vocab files in archive: %s", [os.path.basename(f) for f in vocab_files])

    # Get vocab from decoder weights: decoder.decoder_layers.0.weight shape is [vocab_size, feat_in, 1]
    decoder_key = [k for k in state_dict if 'decoder_layers' in k and 'weight' in k]
    if decoder_key:
        vocab_size = state_dict[decoder_key[0]].shape[0]
        logging.info("Vocab size from decoder weights: %d", vocab_size)

    # For CTC decoding, use the distill.py script approach: load from the training config's teacher model
    # But simpler: use the distill training script to restore, since it knows how to build the student
    # Actually simplest: use the main.py restore which already handles this
    # Let's just use the running training process's model by loading via the distill script's _create_student path

    # Alternative: load the TEACHER model (which we know works) and just use its tokenizer/vocab
    logging.info("Loading teacher model for vocab/tokenizer...")
    from promptingnemo.models.ctc_model import CustomEncDecCTCModelBPE
    teacher = CustomEncDecCTCModelBPE.restore_from(
        '/mnt/training/models/stt_meta_1b/parakeet-600m-encoder-tune-weight-0.5-batch-langfamily-balance-tokenizer-aggregated-new.nemo',
        map_location=device, strict=False,
    )
    vocab = list(teacher.decoder.vocabulary)
    blank_id = len(vocab)
    logging.info("Loaded vocab: %d tokens, blank_id=%d", len(vocab), blank_id)

    # Build a minimal student model for inference using teacher's architecture but student weights
    # The student has encoder_proj (Conv1d 256->1024) + teacher's decoder
    # So we can: run student encoder -> encoder_proj -> teacher decoder
    # But it's simpler to just do raw inference: preprocessor -> encoder -> proj -> decoder -> greedy

    # Extract student components from state_dict
    import torch.nn as nn
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
    from nemo.collections.asr.modules.conv_asr import ConvASRDecoder

    # Build student encoder (d_model=256, 12 layers)
    student_encoder = ConformerEncoder(
        feat_in=80, feat_out=-1, n_layers=12, d_model=256, subsampling='dw_striding',
        subsampling_factor=8, subsampling_conv_channels=256, ff_expansion_factor=4,
        self_attention_model='rel_pos', n_heads=4, conv_kernel_size=15,
        pos_emb_max_len=5000,
    ).to(device)

    # Load encoder weights
    enc_sd = {k.replace('encoder.', '', 1): v for k, v in state_dict.items() if k.startswith('encoder.')}
    student_encoder.load_state_dict(enc_sd, strict=False)
    student_encoder.eval()

    # encoder_proj
    encoder_proj = nn.Conv1d(256, 1024, kernel_size=1).to(device)
    if 'encoder_proj.weight' in state_dict:
        encoder_proj.weight.data.copy_(state_dict['encoder_proj.weight'])
        encoder_proj.bias.data.copy_(state_dict['encoder_proj.bias'])
        logging.info("Loaded encoder_proj weights")
    encoder_proj.eval()

    # Use teacher's preprocessor and decoder (decoder weights are in the student checkpoint)
    preprocessor = teacher.preprocessor
    preprocessor.eval()

    decoder = teacher.decoder
    dec_sd = {k.replace('decoder.', '', 1): v for k, v in state_dict.items() if k.startswith('decoder.')}
    decoder.load_state_dict(dec_sd, strict=False)
    decoder.eval()

    del teacher
    logging.info("Student model assembled: preprocessor -> encoder(256) -> proj(256->1024) -> decoder")
    blank_id = len(vocab)

    # Load manifest
    entries = []
    with open(args.manifest, 'r') as f:
        for line in f:
            e = json.loads(line.strip())
            if 0.5 <= e.get('duration', 0) <= 20.0:
                entries.append(e)
            if len(entries) >= args.n:
                break

    logging.info("Evaluating %d samples", len(entries))

    total_errors = 0
    total_words = 0
    total_errors_with_tags = 0
    total_words_with_tags = 0

    for i, entry in enumerate(entries):
        audio_path = entry['audio_filepath']
        ref_text = entry.get('text', '')
        ref_stripped = strip_tags(ref_text)

        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
        except Exception as e:
            logging.warning("Skip %s: %s", audio_path, e)
            continue

        audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
        audio_len = torch.tensor([len(audio)], dtype=torch.long, device=device)

        with torch.no_grad():
            processed, processed_len = preprocessor(
                input_signal=audio_tensor, length=audio_len
            )
            encoded, encoded_len = student_encoder(
                audio_signal=processed, length=processed_len
            )
            encoded = encoder_proj(encoded)
            log_probs = decoder(encoder_output=encoded)
            greedy_preds = log_probs.argmax(dim=-1, keepdim=False)

        # CTC decode (greedy, collapse repeats + remove blanks)
        pred_ids = greedy_preds[0, :encoded_len[0]].cpu().tolist()
        decoded = []
        prev = None
        for tok_id in pred_ids:
            if tok_id != blank_id and tok_id != prev:
                if tok_id < len(vocab):
                    decoded.append(vocab[tok_id])
            prev = tok_id
        hyp_text = ''.join(decoded).replace('▁', ' ').strip()
        hyp_stripped = strip_tags(hyp_text)

        # WER without tags (correct metric)
        errors, words = word_error_rate(ref_stripped, hyp_stripped)
        total_errors += errors
        total_words += words

        # WER with tags (what the buggy metric was computing)
        errors_wt, words_wt = word_error_rate(ref_text.replace('▁', ' ').strip(),
                                               hyp_text)
        total_errors_with_tags += errors_wt
        total_words_with_tags += words_wt

        if i < 10 or i % 10 == 0:
            logging.info("--- Sample %d ---", i)
            logging.info("  REF:  %s", ref_stripped[:120])
            logging.info("  HYP:  %s", hyp_stripped[:120])
            logging.info("  WER:  %d/%d = %.2f", errors, words, errors / max(words, 1))

    wer_clean = total_errors / max(total_words, 1)
    wer_tags = total_errors_with_tags / max(total_words_with_tags, 1)

    logging.info("========== RESULTS (%d samples) ==========", len(entries))
    logging.info("WER (text only, tags stripped):  %.4f  (%d errors / %d words)", wer_clean, total_errors, total_words)
    logging.info("WER (with tags, buggy metric):   %.4f  (%d errors / %d words)", wer_tags, total_errors_with_tags, total_words_with_tags)
    logging.info("The buggy metric inflates WER because CTC is trained without tags but WER was measured with them.")


if __name__ == '__main__':
    main()
