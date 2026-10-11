"""Hand-written natural voice directions with gold labels, written (2026-10-11) BEFORE any adapter was run on them.

Fields per item: id, cat, prompt,
  spec  {dim: +1/-1}  dims the direction names or unambiguously implies (scored for specification F1 and direction)
  ok    {dim: +1/-1} or "*"  dims that are not penalised if predicted (plausible side effects); "*" = any dim
  tags  acceptable style tags (top-1/top-3 scored only when non-empty)
  neg   {dim: forbidden sign}  negation items: pass iff the dim is not predicted with that sign
  neg_tags  negation items: pass iff the top-1 tag is not in this list
  env   gold for the environment heads (scene / room / phone / ev_start / ev_end), env adapter only
Intensity probes: PROBES = (dim, sign, [weak, plain, strong] prompts); gold = |value| strictly increasing.

python -m steer.paper.adapter_gold  -> paper/results/adapter_handwritten_gold.json
"""
import json

from .common import RES

P, M = 1, -1
ITEMS = [
    # ---- single dimension, both directions (natural phrasings, not the caption generator's)
    ("d01", "dim", "Talk fast, like you're in a hurry.", {"rate": P}, {"pause": M}, []),
    ("d02", "dim", "Take your time and say it slowly.", {"rate": M}, {"pause": P}, []),
    ("d03", "dim", "Say it loud enough for the back row to hear.", {"loud": P}, {"tilt": P, "centroid": P, "cpps": P, "voiced": P}, ["loud"]),
    ("d04", "dim", "Keep your voice down.", {"loud": M}, {"tilt": M, "centroid": M}, []),
    ("d05", "dim", "Use a higher pitch than usual.", {"f0_med": P}, {}, ["high_pitch"]),
    ("d06", "dim", "Drop your voice to a lower pitch.", {"f0_med": M}, {}, ["low_pitch"]),
    ("d07", "dim", "Make the intonation lively and varied.", {"f0_iqr": P}, {"f0_med": P}, []),
    ("d08", "dim", "Read it in a flat monotone.", {"f0_iqr": M}, {}, []),
    ("d09", "dim", "Let your voice go up at the end, as if you're asking.", {"f0_end": P}, {}, []),
    ("d10", "dim", "Finish the sentence with a firm drop in pitch.", {"f0_end": M}, {}, []),
    ("d11", "dim", "Leave long gaps between the phrases.", {"pause": P}, {"rate": M}, []),
    ("d12", "dim", "Say it all in one go, no stopping.", {"pause": M}, {"rate": P}, []),
    ("d13", "dim", "Give your voice a sharp, bright edge.", {"tilt": P}, {"centroid": P, "flat": P}, []),
    ("d14", "dim", "Use a mellow, dark tone.", {"tilt": M}, {"centroid": M, "flat": M}, []),
    ("d15", "dim", "Use your full voice, not breathy at all.", {"voiced": P}, {"flat": M, "cpps": P, "h1h2": M}, []),
    ("d16", "dim", "Whisper it.", {"voiced": M}, {"flat": P, "h1h2": P, "cpps": M, "loud": M, "tilt": P, "centroid": P}, ["whisper"]),
    ("d17", "dim", "Make it sound airy.", {"flat": P}, {"h1h2": P, "voiced": M, "cpps": M}, []),
    ("d18", "dim", "Keep the tone clean and clear, nothing airy about it.", {"flat": M}, {"cpps": P, "voiced": P, "centroid": P}, []),
    ("d19", "dim", "Articulate crisply.", {"centroid": P}, {"tilt": P, "flat": M}, ["enunciated"]),
    ("d20", "dim", "Sound muffled, like talking through a scarf.", {"centroid": M}, {"tilt": M, "loud": M}, []),
    ("d21", "dim", "Keep your voice steady and resonant.", {"cpps": P}, {"voiced": P, "flat": M}, []),
    ("d22", "dim", "Let your voice tremble.", {"cpps": M}, {"voiced": M, "flat": P}, ["fear", "anxiety", "sadness", "distress", "panic"]),
    ("d23", "dim", "A soft, breathy quality.", {"h1h2": P}, {"flat": P, "voiced": M, "loud": M, "cpps": M}, []),
    ("d24", "dim", "Use a pressed, tense voice.", {"h1h2": M}, {"cpps": M, "tilt": P}, []),
    # ---- combinations
    ("c01", "combo", "Speak slowly and quietly.", {"rate": M, "loud": M}, {}, []),
    ("c02", "combo", "Fast and high-pitched.", {"rate": P, "f0_med": P}, {}, []),
    ("c03", "combo", "Low, slow and deliberate.", {"f0_med": M, "rate": M}, {"pause": P}, []),
    ("c04", "combo", "Loud, with the pitch falling at the end.", {"loud": P, "f0_end": M}, {}, []),
    ("c05", "combo", "Whisper it quickly.", {"voiced": M, "rate": P}, {"flat": P, "h1h2": P, "cpps": M, "loud": M}, ["whisper"]),
    ("c06", "combo", "Sad and slow.", {"rate": M}, "*", ["sadness", "disappointment"]),
    ("c07", "combo", "Angry and loud.", {"loud": P}, "*", ["anger", "irritation", "frustration"]),
    ("c08", "combo", "A calm, low voice with plenty of pauses.", {"f0_med": M, "pause": P}, "*", ["serenity", "neutral", "contentment"]),
    # ---- style tags / emotions
    ("t01", "tag", "You're furious about what they did.", {}, "*", ["anger", "irritation", "frustration"]),
    ("t02", "tag", "Say it like you just lost your best friend.", {}, "*", ["sadness", "distress", "pain", "disappointment"]),
    ("t03", "tag", "Sound delighted, this is the best news ever.", {}, "*", ["happiness", "excited", "ecstasy", "amazement"]),
    ("t04", "tag", "You're terrified something is behind you.", {}, "*", ["fear", "panic", "anxiety", "distress"]),
    ("t05", "tag", "Say it like the food smells rotten.", {}, "*", ["disgust", "contempt"]),
    ("t06", "tag", "Sound utterly bored.", {}, "*", ["bored", "tired"]),
    ("t07", "tag", "You're exhausted after a double shift.", {}, "*", ["tired", "bored"]),
    ("t08", "tag", "Say it sarcastically.", {}, "*", ["sarcasm", "contempt", "skeptical"]),
    ("t09", "tag", "Peaceful and soothing, like a meditation guide.", {}, "*", ["serenity", "contentment"]),
    ("t10", "tag", "You can't believe your eyes.", {}, "*", ["amazement", "realization", "confusion"]),
    ("t11", "tag", "Sound puzzled, you don't get it.", {}, "*", ["confusion", "skeptical"]),
    ("t12", "tag", "Say it proudly, you won.", {}, "*", ["pride", "confidence", "happiness", "excited"]),
    ("t13", "tag", "Apologise sincerely.", {}, "*", ["apologetic", "guilt", "embarrassment"]),
    ("t14", "tag", "You're nervous about the interview.", {}, "*", ["anxiety", "fear", "hesitant"]),
    ("t15", "tag", "Laugh while you say it.", {}, "*", ["laughing", "amusement"]),
    ("t16", "tag", "Speak to the baby in a cutesy way.", {}, "*", ["cuteness", "adoration"]),
    # ---- negation
    ("n01", "neg", "Don't speak too fast.", {}, "*", [], {"rate": P}),
    ("n02", "neg", "Not too loud, please.", {}, "*", [], {"loud": P}),
    ("n03", "neg", "Don't whisper; use your normal voice.", {}, "*", [], {"voiced": M}),
    ("n04", "neg", "Don't rush it.", {}, "*", [], {"rate": P}),
    ("n05", "neg", "Never raise your voice.", {}, "*", [], {"loud": P}),
    ("n06", "neg", "Not sad at all, really cheerful.", {}, "*", ["happiness", "excited", "ecstasy", "amusement"], {}, ["sadness", "disappointment"]),
    # ---- the direction buried in a long prompt
    ("l01", "long", "You are reading a bedtime story to a sleepy child in a dim little room, but the book is long, so speak fast.", {"rate": P}, "*", []),
    ("l02", "long", "You are the stadium announcer and the home team has just scored in the final minute: shout it really loudly.", {"loud": P}, "*", []),
    ("l03", "long", "Imagine a late-night radio host closing the show for the few listeners still awake; keep the pitch low.", {"f0_med": M}, "*", []),
    ("l04", "long", "This is a voicemail to your landlord about the broken heating for the third week in a row, and you're out of patience, so say it quickly.", {"rate": P}, "*", ["irritation", "frustration", "anger"]),
    ("l05", "long", "As a museum guide speaking to a group of tourists who keep wandering off, slow down and leave pauses so they can follow.", {"rate": M, "pause": P}, "*", []),
]
ENV_ITEMS = [  # env adapter only; spec/tag gold where a style is also named
    ("e01", "env", "You're in a noisy coffee shop.", dict(scene=["cafe", "restaurant", "crowd"])),
    ("e02", "env", "Like you're calling from your cell phone.", dict(phone=True)),
    ("e03", "env", "In an enormous echoey church.", dict(room=["large"])),
    ("e04", "env", "Say it and laugh at the end.", dict(ev_end=["laugh"])),
    ("e05", "env", "A storm outside, rain hammering on the roof.", dict(scene=["rain", "thunderstorm"])),
    ("e06", "env", "Clear your throat first, then talk.", dict(ev_start=["throat", "cough"])),
    ("e07", "env", "Somewhere by the sea with waves breaking.", dict(scene=["water"])),
    ("e08", "env", "Sigh at the end.", dict(ev_end=["sigh", "breath"])),
    ("e09", "env", "From the back seat of a moving car.", dict(scene=["car", "traffic"])),
    ("e10", "env", "A dog barks and then you speak.", dict(ev_start=["dog"])),
]
PROBE_WORDS = [  # (dim, sign, plain word for "Speak {adv} {w}.")
    ("rate", P, "faster"), ("rate", M, "slower"), ("loud", P, "louder"), ("loud", M, "more quietly"),
    ("f0_med", P, "higher"), ("f0_med", M, "lower"), ("f0_iqr", P, "more expressively"), ("f0_iqr", M, "more monotonously"),
    ("pause", P, "with more pauses"), ("voiced", M, "more breathily"), ("cpps", M, "more shakily"), ("f0_end", P, "with more of a rise at the end"),
]
ADVERBS = [("a little", "", "much"), ("slightly", "", "very much")]


def build():
    items = []
    for it in ITEMS:
        d = dict(id=it[0], cat=it[1], prompt=it[2], spec=it[3], ok=it[4], tags=it[5])
        if len(it) > 6:
            d["neg"] = it[6]
        if len(it) > 7:
            d["neg_tags"] = it[7]
        items.append(d)
    env = [dict(id=i, cat=c, prompt=p, env=g) for i, c, p, g in ENV_ITEMS]
    probes = []
    for dim, sgn, w in PROBE_WORDS:
        for advs in ADVERBS:
            probes.append(dict(dim=dim, sign=sgn, prompts=[" ".join(f"Speak {a} {w}.".split()) for a in advs]))
    return dict(written="2026-10-11, before any model was run on these prompts", items=items, env_items=env,
                probes=probes, gold_rule=__doc__)


if __name__ == "__main__":
    g = build()
    json.dump(g, open(RES / "adapter_handwritten_gold.json", "w"), indent=1)
    print(len(g["items"]), "items,", len(g["env_items"]), "env items,", len(g["probes"]), "probe triples")
