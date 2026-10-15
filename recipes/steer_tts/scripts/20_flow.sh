#!/bin/bash
# Flow-matching decoders on top of the frozen base (run from $STEER_TTS_ROOT: flow paths are relative to it).
#   bash scripts/20_flow.sh <stage>     stages: gta refiner p1 p2 p3 p5
#     gta       teacher-forced ("GTA") base mels of all non-LibriTTS clips + 60k LibriTTS -> data/gta/v3_305k
#     refiner   flow_v3   unprompted OT-CFM refiner, 17.6M, 100k steps          (used for all non-cloning synthesis)
#     p1        flow_v3p  prompted flow (in-clip context infilling), from flow_v3, 60k
#     p2        flow_v3pc + mu_cmn (base-mel cepstral mean removed: voice must come from the prompt), 40k
#     p3        flow_v4pa + zero-gated self-attention at layers 1,3,5,7,9,11; step40000 is the cloning checkpoint
#     p5        flow_v5   steer/pflow5.py: online mu over the full corpus, cross-clip prompts, 30 % voice swap,
#                         RoPE attention + pooled prompt tokens + long skips + speaker-alignment loss
#     eval      UTMOS of L1 vs refiner renders (steer.flow eval)
# Note: gta/refiner/p1-p3 used base step305000 (the then-current checkpoint); synthesis pairs them with the 312.5k
# base. pflow5 runs the 312.5k base online.
source "$(dirname "$0")/env.sh"
cd $W
BASE_GTA=runs/spk_enc_v3/step305000.pt
case "${1:-}" in
gta)     $PY -m steer.flow gta --base $BASE_GTA --out data/gta/v3_305k ;;
refiner) retry 3 $PY -m steer.flow train --gta data/gta/v3_305k --run runs/flow_v3 --steps 100000 ;;
p1)      retry 3 $PY -m steer.flow train --gta data/gta/v3_305k --run runs/flow_v3p --prompt --crop 384 \
           --init runs/flow_v3/step100000.pt --steps 60000 --batch 32 --lr 1e-4 --workers 10 --eval_every 2500 ;;
p2)      retry 3 $PY -m steer.flow train --gta data/gta/v3_305k --run runs/flow_v3pc --prompt --mu_cmn --crop 384 \
           --init runs/flow_v3p/last.pt --steps 40000 --batch 32 --lr 1e-4 --workers 10 --eval_every 2500 ;;
p3)      retry 3 $PY -m steer.flow train --gta data/gta/v3_305k --run runs/flow_v4pa --prompt --mu_cmn --crop 384 \
           --attn_at 1,3,5,7,9,11 --init runs/flow_v3pc/last.pt --lr 1.5e-4 --steps 140000 ;;
p5)      retry 3 $PY -m steer.pflow5 --init runs/flow_v4pa/step40000.pt --run runs/flow_v5 \
           --base runs/spk_enc_v3/step312500.pt --steps 90000 ;;   # auto-resumes from runs/flow_v5/last.pt
eval)    # UTMOS of L1 vs flow renders on quick_eval items
         $PY -m steer.flow eval --flow runs/flow_v3/step100000.pt --base runs/spk_enc_v3/step312500.pt --name v3_flow ;;
*) sed -n 2,14p "$0"; exit 1 ;;
esac
