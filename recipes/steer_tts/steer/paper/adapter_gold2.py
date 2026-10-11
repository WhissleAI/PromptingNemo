"""Second hand-written gold set (~100 relative directions), written 2026-10-11 BLIND to any model output on them,
after reviewer round 1. Same fields as adapter_gold.py, plus:
  grp/lvl  intensity triples: items sharing grp are (weak=0, plain=1, strong=2) versions of one direction;
           gold = |value| on the grp's dim strictly increasing and all three specified with the right sign.
  cat      dim | intensity | neg | combo | distractor | tag
Composition exact match (combo, distractor): every gold dim specified with the right sign and no predicted dim
outside spec+ok.

python -m steer.paper.adapter_gold2 -> paper/results/adapter_handwritten_gold_v2.json
"""
import json

from .common import RES

P, M = 1, -1
ANY = "*"
# (id, cat, prompt, spec, ok, tags[, neg, neg_tags])
ITEMS = [
    # ---- single dims, both directions, new phrasings
    ("v01", "dim", "Pick up the pace a little.", {"rate": P}, {"pause": M}, []),
    ("v02", "dim", "Slow it right down.", {"rate": M}, {"pause": P}, []),
    ("v03", "dim", "Project your voice like you're on stage.", {"loud": P}, {"tilt": P, "centroid": P, "cpps": P, "voiced": P}, ["loud"]),
    ("v04", "dim", "Say it softly, almost under your breath.", {"loud": M}, {"voiced": M, "flat": P, "h1h2": P, "tilt": M, "centroid": M}, ["whisper"]),
    ("v05", "dim", "Raise your pitch.", {"f0_med": P}, {"f0_iqr": P}, ["high_pitch"]),
    ("v06", "dim", "Speak in a deeper register than normal.", {"f0_med": M}, {}, ["low_pitch"]),
    ("v07", "dim", "Put a lot of melody into your intonation.", {"f0_iqr": P}, {"f0_med": P}, []),
    ("v08", "dim", "Keep the pitch completely level, no ups and downs.", {"f0_iqr": M}, {}, []),
    ("v09", "dim", "End on an upturn, like a question.", {"f0_end": P}, {"f0_iqr": P}, []),
    ("v10", "dim", "Bring the pitch down at the very end.", {"f0_end": M}, {}, []),
    ("v11", "dim", "Pause often while you talk.", {"pause": P}, {"rate": M}, ["hesitant"]),
    ("v12", "dim", "Don't leave any gaps, keep it flowing.", {"pause": M}, {"rate": P}, []),
    ("v13", "dim", "Make the voice sound bright and cutting.", {"tilt": P}, {"centroid": P, "loud": P}, []),
    ("v14", "dim", "A warm, soft, dark timbre.", {"tilt": M}, {"centroid": M, "loud": M, "h1h2": P}, []),
    ("v15", "dim", "Fully voiced, no whispering at all.", {"voiced": P}, {"flat": M, "cpps": P, "loud": P}, []),
    ("v16", "dim", "Use a hushed whisper.", {"voiced": M}, {"loud": M, "flat": P, "h1h2": P, "cpps": M, "tilt": P, "centroid": P}, ["whisper"]),
    ("v17", "dim", "Let a lot of breath into the voice.", {"flat": P}, {"h1h2": P, "voiced": M, "cpps": M}, []),
    ("v18", "dim", "A clean, focused tone without breathiness.", {"flat": M}, {"cpps": P, "voiced": P, "h1h2": M}, []),
    ("v19", "dim", "Make it sound brighter and crisper.", {"centroid": P}, {"tilt": P}, ["enunciated"]),
    ("v20", "dim", "Make it sound dull and muffled.", {"centroid": M}, {"tilt": M, "loud": M}, []),
    ("v21", "dim", "A clear, ringing, stable voice.", {"cpps": P}, {"voiced": P, "flat": M, "centroid": P, "tilt": P}, []),
    ("v22", "dim", "A rough, wobbly, unsteady voice.", {"cpps": M}, {"flat": P, "voiced": M}, []),
    ("v23", "dim", "Give it a gentle, breathy softness.", {"h1h2": P}, {"flat": P, "loud": M, "voiced": M, "cpps": M}, []),
    ("v24", "dim", "Squeeze the voice so it sounds tight and pressed.", {"h1h2": M}, {"tilt": P, "cpps": M}, []),
    # ---- intensity triples (grp, lvl) appended below
    # ---- negation
    ("g01", "neg", "Not too fast.", {}, ANY, [], {"rate": P}),
    ("g02", "neg", "Don't shout.", {}, ANY, [], {"loud": P}),
    ("g03", "neg", "No whispering, please.", {}, ANY, [], {"voiced": M}),
    ("g04", "neg", "Don't be monotone.", {}, ANY, [], {"f0_iqr": M}),
    ("g05", "neg", "Don't speed up.", {}, ANY, [], {"rate": P}),
    ("g06", "neg", "Don't drop your pitch.", {}, ANY, [], {"f0_med": M}),
    ("g07", "neg", "No long pauses.", {}, ANY, [], {"pause": P}),
    ("g08", "neg", "Don't slow down.", {}, ANY, [], {"rate": M}),
    ("g09", "neg", "Never sound sad here.", {}, ANY, [], {}, ["sadness", "disappointment"]),
    ("g10", "neg", "Not angry, just calm.", {}, ANY, ["serenity", "neutral", "contentment"], {}, ["anger", "irritation", "frustration"]),
    ("g11", "neg", "Don't make it quiet.", {}, ANY, [], {"loud": M}),
    ("g12", "neg", "Avoid a high, squeaky pitch.", {}, ANY, [], {"f0_med": P}),
    # ---- composition (2-3 attributes)
    ("k01", "combo", "Fast and loud.", {"rate": P, "loud": P}, {"tilt": P, "centroid": P}, []),
    ("k02", "combo", "Slow and low-pitched.", {"rate": M, "f0_med": M}, {"pause": P}, []),
    ("k03", "combo", "Quiet, slow, and with long pauses.", {"loud": M, "rate": M, "pause": P}, {}, []),
    ("k04", "combo", "High-pitched and lively intonation.", {"f0_med": P, "f0_iqr": P}, {}, []),
    ("k05", "combo", "Loud and fast, ending with a rising pitch.", {"loud": P, "rate": P, "f0_end": P}, {"tilt": P, "centroid": P}, []),
    ("k06", "combo", "Breathy and quiet.", {"flat": P, "loud": M}, {"h1h2": P, "voiced": M, "cpps": M}, []),
    ("k07", "combo", "A low, monotone voice.", {"f0_med": M, "f0_iqr": M}, {}, []),
    ("k08", "combo", "Quick, with no pauses at all.", {"rate": P, "pause": M}, {}, []),
    ("k09", "combo", "Slower, quieter and lower.", {"rate": M, "loud": M, "f0_med": M}, {"pause": P}, []),
    ("k10", "combo", "Louder, with a falling ending.", {"loud": P, "f0_end": M}, {"tilt": P}, []),
    ("k11", "combo", "Higher and faster.", {"f0_med": P, "rate": P}, {}, []),
    ("k12", "combo", "Soft and breathy, with a mellow tone.", {"loud": M, "h1h2": P, "tilt": M}, {"flat": P, "voiced": M, "centroid": M}, []),
    ("k13", "combo", "Happy and fast.", {"rate": P}, ANY, ["happiness", "excited", "ecstasy"]),
    ("k14", "combo", "Scared and whispering.", {"voiced": M}, ANY, ["fear", "whisper", "anxiety", "panic"]),
    ("k15", "combo", "Angry, loud and fast.", {"loud": P, "rate": P}, ANY, ["anger", "irritation", "frustration"]),
    ("k16", "combo", "Sad, slow and quiet.", {"rate": M, "loud": M}, ANY, ["sadness", "disappointment"]),
    # ---- distractors (content words that point the other way)
    ("x01", "distractor", "Read this line about a fast car slowly.", {"rate": M}, {"pause": P}, []),
    ("x02", "distractor", "Describe the loud concert quietly.", {"loud": M}, {"tilt": M, "centroid": M}, []),
    ("x03", "distractor", "Talk about the high mountains in a low voice.", {"f0_med": M}, {}, []),
    ("x04", "distractor", "A story about a slow turtle, read quickly.", {"rate": P}, {"pause": M}, []),
    ("x05", "distractor", "Say 'I'm so happy' in a sad voice.", {}, ANY, ["sadness", "disappointment"]),
    ("x06", "distractor", "The word whisper is in the text, but say it loudly.", {"loud": P}, {"tilt": P, "centroid": P, "voiced": P}, ["loud"]),
    ("x07", "distractor", "Calmly explain why the alarm is so loud.", {}, ANY, ["serenity", "neutral", "contentment"]),
    ("x08", "distractor", "Read the angry customer's email in a cheerful voice.", {}, ANY, ["happiness", "excited"]),
    ("x09", "distractor", "Speak quickly about the quiet library.", {"rate": P}, {"pause": M}, []),
    ("x10", "distractor", "In a deep voice, describe the high-pitched whistle.", {"f0_med": M}, {}, ["low_pitch"]),
    # ---- tags
    ("y01", "tag", "Sound thrilled to bits.", {}, ANY, ["excited", "happiness", "ecstasy"]),
    ("y02", "tag", "Sound heartbroken.", {}, ANY, ["sadness", "distress", "pain", "disappointment"]),
    ("y03", "tag", "You're seething with rage.", {}, ANY, ["anger", "irritation", "frustration"]),
    ("y04", "tag", "You're scared out of your mind.", {}, ANY, ["fear", "panic", "anxiety", "distress"]),
    ("y05", "tag", "You find this repulsive.", {}, ANY, ["disgust", "contempt"]),
    ("y06", "tag", "Sound completely uninterested.", {}, ANY, ["bored", "tired"]),
    ("y07", "tag", "You're half asleep.", {}, ANY, ["tired", "bored"]),
    ("y08", "tag", "Say it with heavy irony.", {}, ANY, ["sarcasm", "contempt", "skeptical"]),
    ("y09", "tag", "Relaxed and tranquil.", {}, ANY, ["serenity", "contentment"]),
    ("y10", "tag", "Totally astonished.", {}, ANY, ["amazement", "realization"]),
    ("y11", "tag", "Unsure and a bit lost.", {}, ANY, ["confusion", "hesitant", "skeptical"]),
    ("y12", "tag", "Full of confidence.", {}, ANY, ["confidence", "pride", "determined"]),
    ("y13", "tag", "Sound ashamed of yourself.", {}, ANY, ["guilt", "embarrassment", "apologetic"]),
    ("y14", "tag", "You're doubtful this will work.", {}, ANY, ["skeptical", "hesitant", "confusion"]),
    ("y15", "tag", "Say it with a giggle.", {}, ANY, ["laughing", "amusement"]),
    ("y16", "tag", "Tenderly, like to someone you love.", {}, ANY, ["adoration", "desire", "sympathy"]),
]
# intensity triples: (dim, sign, weak, plain, strong), new constructions (not "Speak {adv} {word}.")
TRIPLES = [
    ("rate", P, "Talk a touch quicker.", "Talk quicker.", "Talk way, way quicker."),
    ("rate", M, "Ease off the pace just slightly.", "Ease off the pace.", "Ease off the pace dramatically."),
    ("loud", P, "Be a tiny bit louder.", "Be louder.", "Be extremely loud."),
    ("loud", M, "Lower your volume a little.", "Lower your volume.", "Lower your volume a lot."),
    ("f0_med", P, "Pitch it slightly higher.", "Pitch it higher.", "Pitch it much, much higher."),
    ("f0_med", M, "A slightly deeper voice.", "A deeper voice.", "A far deeper voice."),
    ("f0_iqr", P, "A bit more intonation.", "More intonation.", "Hugely more intonation."),
    ("pause", P, "A few more pauses than usual.", "More pauses than usual.", "Far more pauses than usual."),
]
OKS = {"rate": {"pause": M}, "loud": {"tilt": P, "centroid": P}, "f0_med": {"f0_iqr": P}, "f0_iqr": {"f0_med": P}, "pause": {"rate": M}}


def build():
    items = []
    for it in ITEMS:
        d = dict(id=it[0], cat=it[1], prompt=it[2], spec=it[3], ok=it[4], tags=it[5])
        if len(it) > 6:
            d["neg"] = it[6]
        if len(it) > 7:
            d["neg_tags"] = it[7]
        items.append(d)
    for g, (dim, sg, *ps) in enumerate(TRIPLES):
        ok = dict(OKS.get(dim, {}))
        if dim == "loud" and sg == M:
            ok = {"tilt": M, "centroid": M}
        for lvl, p in enumerate(ps):
            items.append(dict(id=f"i{g + 1:02d}{'wps'[lvl]}", cat="intensity", prompt=p, spec={dim: sg}, ok=ok, tags=[],
                              grp=g, lvl=lvl))
    return dict(written="2026-10-11, after reviewer round 1, blind to model outputs on these prompts", items=items,
                gold_rule=__doc__)


if __name__ == "__main__":
    g = build()
    json.dump(g, open(RES / "adapter_handwritten_gold_v2.json", "w"), indent=1)
    from collections import Counter
    print(len(g["items"]), Counter(i["cat"] for i in g["items"]))
