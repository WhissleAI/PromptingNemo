"""UTMOSv2 (Baba et al., sarulab-speech/UTMOSv2, fusion_stage3 fold0, pretrained=True) on a list of wavs.

~/anaconda3/envs/tb_utmosv2/bin/python -s steer/paper/utmosv2_score.py <list.txt> <cache.json> [cuda|cpu]
Incremental cache {path: mos}. Not used in any SteerTTS training (unlike UTMOS22). GPU ~1-2 GB.
"""
import json
import os
import sys


def main():
    lst, cache = sys.argv[1], sys.argv[2]
    dev = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    import torch
    import utmosv2
    torch.set_num_threads(4)
    res = json.load(open(cache)) if os.path.exists(cache) else {}
    paths = [l.strip() for l in open(lst) if l.strip()]
    todo = [p for p in paths if p not in res and os.path.exists(p)]
    print(f"utmosv2: {len(todo)} new of {len(paths)} on {dev}", flush=True)
    if not todo:
        return
    m = utmosv2.create_model(pretrained=True, device=dev)
    for i, p in enumerate(todo):
        res[p] = float(m.predict(input_path=p, device=dev, num_workers=0, verbose=False))
        if i % 500 == 499:
            json.dump(res, open(cache, "w"))
            print(f"  {i + 1}/{len(todo)}", flush=True)
    json.dump(res, open(cache, "w"))
    print("utmosv2 done", len(res), flush=True)


if __name__ == "__main__":
    main()
