## INSTRUCTION SPECIFIC ASR+NL MODEL TRAINING SCRIPT

ARTIFACTS_DIR="/working_dir/artifacts/"
# Assigning the name for the process and create model dir
NAME="1step_Conformer_EN_CommonVoice_text-ner-pos_round2"
mkdir -p "$ARTIFACTS_DIR/$NAME"

# Defining the prompt that will be used for filtering the data
INSTRUCTION="Transcribe, mark named entities and track speaker emotion"



# Specifying the full training manifest file path
MANIFEST_DIR="/audio_datasets/manifests/"
MANIFEST_TRAIN_FULL="$MANIFEST_DIR/train_cv_en_round2_withdur.json"
MANIFEST_TRAIN_SELECTED="$MANIFEST_DIR/${NAME}_train_round2.json"

# Extract and process the desired text from the manifest
grep "$INSTRUCTION" "$MANIFEST_TRAIN_FULL" > "$MANIFEST_TRAIN_SELECTED"

# Conditionally executing the learn_tokenizer.py script
LEARN_TOKENIZER=false
VOCAB_SIZE=1024

# Extract and process the text to create a taglist
USER_DEFINED_SYMBOLS_FILE="$MANIFEST_DIR/taglist_train_cv_en.txt"
cat "$MANIFEST_TRAIN_FULL" | jq -r '.text' | grep -o -E '\b(EMOTION|POS|NER)[A-Z_]*\b|\bEND\b' | sort | uniq > "$USER_DEFINED_SYMBOLS_FILE"


if [ "$LEARN_TOKENIZER" = true ]; then
    TOKENIZER_PATH="$ARTIFACTS_DIR/$NAME/tokenizer_spe_bpe_v${VOCAB_SIZE}"
    python ./scripts/nemo/process_asr_text_tokenizer.py \
    --manifest $MANIFEST_TRAIN_FULL \
    --data_root $TOKENIZER_PATH \
    --vocab_size $VOCAB_SIZE \
    --user_defined_symbols_file $USER_DEFINED_SYMBOLS_FILE \
    --tokenizer="spe" \
    --no_lower_case
else
    # Handle the case when LEARN_TOKENIZER is false
    # Define an alternate path or command here
    TOKENIZER_PATH="/working_dir/artifacts/1step_Conformer_EN_CommonVoice_text-ner-pos_round2/tokenizer_spe_bpe_v1024"
    #python add_new_tags $TOKENIZER_PATH $MANIFEST_TRAIN_SELECTED
    #NEW_TOKENIZER_PATH="$TOKENIZER_PATH"
    # You can add any commands or actions to be executed when LEARN_TOKENIZER is false here
    echo "LEARN_TOKENIZER is set to false. Using alternate path: $TOKENIZER_PATH"
fi

## fetch tokenizer from S3 tokenizer-bucket: 
# sentence piece library
TOKENIZER_PATH="${TOKENIZER_PATH}/tokenizer_spe_bpe_v${VOCAB_SIZE}"



# Specifying the full development manifest file path
MANIFEST_DEV_FULL="/audio_datasets/manifests/dev_cv_en.json"
MANIFEST_DEV_SELECTED="/audio_datasets/manifests/${NAME}_dev.json"
grep "$INSTRUCTION" "$MANIFEST_DEV_FULL" > "$MANIFEST_DEV_SELECTED"


python ./scripts/nemo/speech_to_text_bpe.py \
 --config-path="/workspace/nemo/examples/asr/conf/conformer" \
 --config-name="conformer_ctc_bpe.yaml" \
 model.tokenizer.dir="$TOKENIZER_PATH" \
 model.tokenizer.type="bpe" \
 model.encoder.subsampling="dw_striding" \
 model.encoder.subsampling_factor=8 \
 model.encoder.subsampling_conv_channels=256 \
 model.encoder.conv_kernel_size=9 \
 model.train_ds.manifest_filepath="$MANIFEST_TRAIN_SELECTED" \
 model.train_ds.batch_size=12 \
 model.validation_ds.manifest_filepath="$MANIFEST_DEV_SELECTED" \
 model.validation_ds.batch_size=12 \
 model.optim.lr=1.0 \
 model.optim.weight_decay=0.001 \
 trainer.devices=-1 \
 trainer.max_epochs=500 \
 trainer.accumulate_grad_batches=8 \
 trainer.log_every_n_steps=100 \
 ++exp_manager.checkpoint_callback_params.always_save_nemo=True \
 ++exp_manager.checkpoint_callback_params.save_top_k=100 \
 +init_from_nemo_model=/working_dir/artifacts/1step_Conformer_EN_CommonVoice_text-ner-pos_round2/Conformer-CTC-BPE/2023-11-26_01-56-32/checkpoints/Conformer-CTC-BPE.nemo \
 ++exp_manager.exp_dir="$ARTIFACTS_DIR/$NAME"



