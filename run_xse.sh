python /n/disk1/NeMo/examples/nlp/token_classification/punctuation_capitalization_lexical_audio_train_evaluate.pyr \
       model.train_ds.ds_item=/n/disk1/prompting_nemo_datasets/data/punc_restore \
       model.train_ds.text_file=text_dev.txt \
       model.train_ds.labels_file=labels_dev.txt \
       model.validation_ds.ds_item=/n/disk1/prompting_nemo_datasets/data/punc_restore \
       model.validation_ds.text_file=text_dev.txt \
       model.validation_ds.labels_file=labels_dev.txt \
       trainer.devices=[0,1] \
       trainer.accelerator='gpu' \
       optim.name=adam \
       optim.lr=0.0001 \
       model.train_ds.audio_file=audio_dev.txt \
       model.validation_ds.audio_file=audio_dev.txt

