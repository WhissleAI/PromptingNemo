### Fine-tune nemo based asr models




#### Audio only

setup config.yml
```
model:
  model_root: "/external2/models/hf/stt_en_conformer_ctc_large/"
  model_name: "stt_en_conformer_ctc_large.nemo"
  tokenizer_folder: "tokenizer"
  new_tokenizer_folder: "new_tokenizer"
  proto_file: "proto/file"
  proto_dir: "proto/dir"

training:
  data_dir: "/external2/datasets/slurp"
  train_manifest: "train-slurp-tagged.json"
  test_manifest: "devel-slurp-tagged.json"
  batch_size: 16
  max_steps: 600000
```

run fine-tuning
```
python nemo_adapter.py
```




#### Audio-visual Only

you will need Whissle's fork of NeMo. 