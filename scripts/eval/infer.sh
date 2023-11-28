
CKP=$1
INP=$2
OUT=$3

python ./ASR-NL/scripts/nemo/transcribe_speech.py model_path=$CKP dataset_manifest=$INP output_filename=$OUT
