#!/bin/bash
# Downloads raw corpora for steer_tts. Re-runnable: skips finished files.
set -u
R="${STEER_TTS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}/data/raw"
hf() {  # repo path dest
  mkdir -p "$(dirname "$3")"; [ -s "$3" ] && return 0
  curl -sfL --retry 5 "https://huggingface.co/datasets/$1/resolve/main/$2" -o "$3.part" && mv "$3.part" "$3" || echo "FAIL $1/$2"
}
case "$1" in
libritts)
  curl -s "https://huggingface.co/api/datasets/mythicinfinity/libritts_r/tree/main/data/train.clean.100" | python3 -c "import json,sys;[print(x['path']) for x in json.load(sys.stdin)]" > $R/libritts_r/files.txt
  curl -s "https://huggingface.co/api/datasets/mythicinfinity/libritts_r/tree/main/data/dev.clean" | python3 -c "import json,sys;[print(x['path']) for x in json.load(sys.stdin)]" >> $R/libritts_r/files.txt
  while read f; do hf mythicinfinity/libritts_r "$f" "$R/libritts_r/$f"; done < $R/libritts_r/files.txt
  hf parler-tts/libritts-r-filtered-speaker-descriptions clean/train.clean.100-00000-of-00001.parquet $R/libritts_r/parler_desc_train.clean.100.parquet ;;
expresso)
  curl -s "https://huggingface.co/api/datasets/ylacombe/expresso/tree/main/read" | python3 -c "import json,sys;[print(x['path']) for x in json.load(sys.stdin)]" > $R/expresso/files.txt
  while read f; do hf ylacombe/expresso "$f" "$R/expresso/$f"; done < $R/expresso/files.txt ;;
ears)
  cd $R/ears
  curl -sfL https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/transcripts.json -o transcripts.json
  curl -sfL https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/speaker_statistics.json -o speaker_statistics.json
  for X in $(seq -f "%03g" 1 107); do
    [ -d p$X ] && continue
    curl -sfL --retry 5 https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p$X.zip -o p$X.zip.part && mv p$X.zip.part p$X.zip && unzip -q p$X.zip && rm p$X.zip || echo "FAIL p$X"
  done ;;
esac
echo "DONE $1"
