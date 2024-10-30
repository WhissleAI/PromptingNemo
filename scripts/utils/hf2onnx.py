from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_checkpoint = "DAMO-NLP-SG/zero-shot-classify-SSTuning-base"
model_shelf =  "/external/ksingla/artifacts/model_shelf"
save_directory = Path(model_shelf) / "zero-shot-classify-SSTuning-base"

# Load a model from transformers and export it to ONNX
ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Save the onnx model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

config = {
    "task": "text_classification_zeroshot",
    "hf_id": "DAMO-NLP-SG/zero-shot-classify-SSTuning-base",
    "sample_rate": 16000,
    "encoder.onnx": "model.onnx",
    "tokenizer.model": "tokenizer/tokenizer.model",
    "onnx.intra_op_num_threads": 1
}

# Convert dictionary to plain text format
config_text = "\n".join(f"{key}={value}" for key, value in config.items()) + "\n"

# Write the plain text to magic.txt
magic_file = open(save_directory / "magic.txt",'w')
magic_file.write(config_text)
magic_file.close()
