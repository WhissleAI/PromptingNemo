#!/bin/bash
# Raw corpora -> $STEER_TTS_ROOT/data/raw. Network/disk only; every step is resumable (re-run until complete).
#   bash scripts/01_download.sh [ears expresso libritts libritts360 libritts500 nvtts noise]
# Licences: EARS, Expresso CC BY-NC 4.0; LibriTTS-R CC BY 4.0; ParaSpeechCaps CC BY-NC-SA 4.0; NonverbalTTS (incl.
# VoxCeleb / Expresso audio) per its card; MUSAN CC BY 4.0, RIRS_NOISES Apache 2.0, DEMAND CC BY-SA 3.0.
# Cartesia renders are generated, not downloaded: see steer/cartesia_gen.py (paid API, key in ~/.config/cartesia/api_key).
# NOTE: steer/prep.py (nvtts) and steer/acoustic.py hard-code the HDD path /external1/karan/steer_tts/noise_raw for
# NonverbalTTS and the noise archives; NOISE_RAW defaults to it. Elsewhere, symlink that path to your NOISE_RAW.
source "$(dirname "$0")/env.sh"
NOISE_RAW=${NOISE_RAW:-/external1/karan/steer_tts/noise_raw}
D=$W/data/raw; mkdir -p $D/ears $D/expresso $D/libritts_r
snap() {  # repo pattern localdir
  $PY -c "from huggingface_hub import snapshot_download as s; s(repo_id='$1', repo_type='dataset', allow_patterns=['$2'], local_dir='$3', max_workers=3)"
}
for c in ${@:-ears expresso libritts libritts360 libritts500 nvtts}; do
  echo "=== $c $(date +%T)"
  case $c in
    ears|expresso|libritts) bash $RECIPE/data/download.sh $c ;;   # EARS 107 spk (GitHub), Expresso read, LibriTTS-R clean.100
    libritts360) snap mythicinfinity/libritts_r 'data/train.clean.360/*' $D/libritts_r ;;
    libritts500) snap mythicinfinity/libritts_r 'data/train.other.500/*' $D/libritts_r ;;
    nvtts)       snap deepvk/NonverbalTTS '*' $NOISE_RAW/nonverbaltts ;;
    paraspeechcaps) bash $RECIPE/data/hf_curl.sh $RECIPE/data/hf_files.tsv ;;   # + PSC; needs data/.hf_curlrc
    noise)  # scene / room sources for steer.acoustic (MUSAN, RIRS_NOISES, DEMAND); large, put on a big disk
      mkdir -p $NOISE_RAW && cd $NOISE_RAW
      get() { [ -f "$2.done" ] || { wget -q -c -O "$2" "$1" && touch "$2.done"; }; }
      get https://www.openslr.org/resources/28/rirs_noises.zip rirs_noises.zip
      mkdir -p rirs && [ -f rirs/.ok ] || { unzip -q -o rirs_noises.zip -d rirs && touch rirs/.ok; }
      for e in DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO; do
        get https://zenodo.org/records/1227121/files/${e}_48k.zip ${e}_48k.zip
      done
      mkdir -p demand && for z in *_48k.zip; do unzip -q -n $z -d demand; done
      get https://www.openslr.org/resources/17/musan.tar.gz musan.tar.gz
      mkdir -p musan_x && [ -f musan_x/.ok ] || { tar xzf musan.tar.gz -C musan_x && touch musan_x/.ok; }
      cd - > /dev/null ;;
    *) echo "unknown corpus $c"; exit 1 ;;
  esac
done
echo "DOWNLOAD DONE $(date)"
