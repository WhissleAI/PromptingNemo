"""SentencePiece tokenizer training utilities."""

import json
import logging
import os
import re

import sentencepiece as spm


def train_sentencepiece_tokenizer(
    manifest_file,
    tokenizer_folder,
    special_tokens=None,
    vocab_size=5000,
    character_coverage=None,
    extra_options=None,
):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting the tokenizer training process")
    logging.info(f"Requested vocab_size={vocab_size}")

    # Step 1: Read the manifest file and extract text data
    def read_manifest(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        return [json.loads(line)['text'] for line in lines]

    logging.info("Reading manifest file")
    text_data = read_manifest(manifest_file)
    logging.info(f"Extracted {len(text_data)} sentences from the manifest file")

    # Step 2: Save the extracted text to a temporary file
    if not os.path.exists(tokenizer_folder):
        os.makedirs(tokenizer_folder)

    temp_text_file = os.path.join(tokenizer_folder, 'text_data.txt')
    logging.info(f"Saving extracted text to {temp_text_file}")
    with open(temp_text_file, 'w') as f:
        for sentence in text_data:
            f.write(sentence + '\n')

    # Step 3: Train the SentencePiece tokenizer with special tokens if provided
    model_prefix = os.path.join(tokenizer_folder, 'tokenizer')

    # Prepare special tokens string
    special_tokens = special_tokens or []
    filtered_special_tokens = [token for token in special_tokens if token and token.strip()]
    logging.info(f"Using {len(filtered_special_tokens)} special tokens")

    # Check if there are any duplicate tokens after case normalization
    token_set = set()
    unique_tokens = []
    for token in filtered_special_tokens:
        token_upper = token.upper()
        if token_upper not in token_set:
            token_set.add(token_upper)
            unique_tokens.append(token_upper)
        else:
            logging.warning(f"Duplicate token after case normalization: {token}")

    print("\n\nMY UNIQUE TOKENS\n\n")
    print(unique_tokens)

    user_defined_symbols = ','.join(unique_tokens) if unique_tokens else ''

    # Retry loop to adapt vocabulary size when SentencePiece complains
    sp_train_kwargs = {
        'input': temp_text_file,
        'model_prefix': model_prefix,
        'max_sentence_length': 8000,
    }
    if character_coverage is not None:
        sp_train_kwargs['character_coverage'] = character_coverage
    if extra_options:
        sp_train_kwargs.update(extra_options)
    if user_defined_symbols:
        sp_train_kwargs['user_defined_symbols'] = user_defined_symbols

    current_vocab_size = vocab_size
    min_vocab_size = len(unique_tokens) + 1  # safety guard
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        logging.info(f"[Attempt {attempt}/{max_attempts}] Training SentencePiece with vocab_size={current_vocab_size}")
        try:
            spm.SentencePieceTrainer.train(vocab_size=current_vocab_size, **sp_train_kwargs)
            break
        except RuntimeError as exc:
            msg = str(exc)
            logging.error(f"SentencePiece training failed: {msg}")
            # Too small
            match_small = re.search(r"Vocabulary size is smaller than required_chars\. (\d+) vs (\d+)", msg)
            if match_small:
                requested = int(match_small.group(1))
                required = int(match_small.group(2))
                current_vocab_size = max(required + 1, current_vocab_size * 2)
                logging.info(f"Increasing vocab size from {requested} to {current_vocab_size}")
                continue
            # Too large
            match_large = re.search(r"Vocabulary size too high \((\d+)\).*value <= (\d+)", msg)
            if match_large:
                requested = int(match_large.group(1))
                limit = int(match_large.group(2))
                current_vocab_size = max(min(limit, current_vocab_size - 1), min_vocab_size)
                logging.info(f"Reducing vocab size from {requested} to {current_vocab_size}")
                continue
            # Unknown error -- re-raise
            raise
    else:
        raise RuntimeError("SentencePiece training failed after multiple attempts; see logs for details.")

    # Step 4: Return the paths to the tokenizer model and vocab files
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    logging.info(f"Tokenizer training completed")
    logging.info(f"Model file: {model_file}")
    logging.info(f"Vocab file: {vocab_file}")

    # Step 5: Create a vocab.txt file
    vocab_txt_file = os.path.join(tokenizer_folder, 'vocab.txt')
    logging.info(f"Creating vocab.txt file at {vocab_txt_file}")
    with open(vocab_file, 'r') as vf, open(vocab_txt_file, 'w') as vtf:
        for line in vf:
            token = line.split('\t')[0]
            vtf.write(token + '\n')

    logging.info(f"vocab.txt file created at {vocab_txt_file}")

    return model_file, vocab_file, vocab_txt_file
