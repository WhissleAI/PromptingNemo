python ./scripts/nemo/speech_to_text_bpe.py \
 --config-path="/workspace/nemo/examples/asr/conf/conformer" \
 --config-name="conformer_ctc_bpe.yaml" \
 model.tokenizer.dir="/ksingla/data_capture/smart_transcription/two-pass-redaction/data/fisherswd/tokenizers/tokenizer_spe_bpe_v1024" \
 model.tokenizer.type="bpe" \
 model.encoder.subsampling="dw_striding" \
 model.encoder.subsampling_factor=8 \
 model.encoder.subsampling_conv_channels=256 \
 model.encoder.conv_kernel_size=9 \
 model.train_ds.manifest_filepath="/ksingla/data_capture/smart_transcription/two-pass-redaction/data/fisherswd/train_digits.json" \
 model.train_ds.batch_size=32 \
 model.validation_ds.manifest_filepath="/ksingla/data_capture/smart_transcription/two-pass-redaction/data/fisherswd/valid_digits.json" \
 model.validation_ds.batch_size=32 \
 model.optim.lr=1.0 \
 model.optim.weight_decay=0.001 \
 trainer.devices=-1 \
 trainer.max_epochs=100 \
 trainer.accumulate_grad_batches=8 \
 trainer.log_every_n_steps=100 \
 ++exp_manager.checkpoint_callback_params.always_save_nemo=True \
 ++exp_manager.checkpoint_callback_params.save_top_k=100 \
 +init_from_nemo_model=/ksingla/data_capture/smart_transcription/two-pass-redaction/pretrained/Conformer-CTC-BPE_caware.nemo \
 ++exp_manager.exp_dir="/ksingla/data_capture/smart_transcription/two-pass-redaction/models/conformer_fisherswd_digits_caware"



