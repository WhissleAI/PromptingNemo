#!/bin/bash
# Seed-TTS-eval test-en zero-shot cloning with the benchmark's own scorers (Whisper-large-v3 WER, WavLM-large SV SIM).
#   SEEDTTS_DIR=... SEEDEVAL_PY=... bash scripts/31_seedtts.sh <name> <ckpt-spec>
# SEEDTTS_DIR holds: seed-tts-eval/ (repo), data/seedtts_testset/en/meta.lst, wavlm_large_finetune.pth, and sv_runner/
# = a copy of seed-tts-eval's speaker_verification/ WITHOUT its select.py (that file shadows the stdlib `select` module
# and crashes on python 3.10). SEEDEVAL_PY is the scoring env's python (torch/torchaudio 2.7.1: s3prl needs
# torchaudio.set_audio_backend). SEED_FP16=1 uses run_wer_fp16.py (= run_wer.py with Whisper in fp16, to fit beside
# training); the default is the unmodified run_wer.py.
# Results: runs/seedtts/<name>/{wer.score,sim.score}
source "$(dirname "$0")/env.sh"
: "${SEEDTTS_DIR:?}" "${SEEDEVAL_PY:?}"
NAME=$1; CKPT=$2; S=$SEEDTTS_DIR
META=$(ls $S/data/*/en/meta.lst | head -1)
OUT=$W/runs/seedtts/$NAME
set -e
(cd $W && $PY -m steer.seedtts --ckpt $CKPT --meta $META --out $OUT)
cd $S/seed-tts-eval
$SEEDEVAL_PY get_wav_res_ref_text.py $META $OUT $OUT/wav_res_ref_text
WER_PY=run_wer.py; [ "${SEED_FP16:-0}" = 1 ] && WER_PY=run_wer_fp16.py
$SEEDEVAL_PY $WER_PY $OUT/wav_res_ref_text $OUT/wer.out en
$SEEDEVAL_PY average_wer.py $OUT/wer.out $OUT/wer.score
cd $S/sv_runner
$SEEDEVAL_PY verification_pair_list_v2.py $OUT/wav_res_ref_text --model_name wavlm_large \
  --checkpoint $S/wavlm_large_finetune.pth --scores $OUT/sim.out \
  --wav1_start_sr 0 --wav2_start_sr 0 --wav1_end_sr -1 --wav2_end_sr -1 --device cuda:0
grep -v "avg score" $OUT/sim.out > $OUT/sim.merge
$SEEDEVAL_PY average.py $OUT/sim.merge $OUT/sim.score
echo "$NAME: WER $(tr '\n' ' ' < $OUT/wer.score) | SIM $(tr '\n' ' ' < $OUT/sim.score)"
(cd $RECIPE && $PY -m steer.tb_report > /dev/null)   # tables -> TensorBoard TEXT tab
