"""emotion2vec+ large scorer (standalone; run in the `seedeval` env, which has funasr).
Produced: called by x3_tags_heldout --stage score (writes runs/paper/x3/e2v.jsonl). All commands run from $STEER_TTS_ROOT with PYTHONPATH=PromptingNemo/recipes/steer_tts.

~/anaconda3/envs/seedeval/bin/python e2v_score.py <list.txt (one wav path per line)> <out.jsonl>
Writes {path, labels, scores} per clip (9 classes: angry disgusted fearful happy neutral other sad surprised unknown).
"""
import json
import sys
from pathlib import Path

from funasr import AutoModel

inp, out = sys.argv[1], sys.argv[2]
paths = [l.strip() for l in open(inp) if l.strip()]
done = set()
if Path(out).exists():
    done = {json.loads(l)["path"] for l in open(out)}
todo = [p for p in paths if p not in done]
m = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True)
with open(out, "a") as f:
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    for i in range(0, len(todo), B):
        ch = todo[i:i + B]
        res = m.generate(ch, granularity="utterance", extract_embedding=False, batch_size=len(ch))
        for p, r in zip(ch, res):
            labels = [l.split("/")[-1] for l in r["labels"]]
            f.write(json.dumps(dict(path=p, labels=labels, scores=[float(s) for s in r["scores"]])) + "\n")
        if (i // B) % 20 == 0:
            print(f"e2v {i + len(ch)}/{len(todo)}", flush=True)
print("E2V DONE", len(paths))
