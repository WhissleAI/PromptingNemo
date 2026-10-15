"""Measured controls + style tag -> natural-language voice directions (training data for the prompt adapter).

Each caption is anchored in acoustics: a phrase about a dimension is only emitted when that utterance's
speaker-relative z on that dimension is salient, and the adverb grade follows |z|. The adapter therefore
learns prompt -> (tag, which dims, how much) from real measurements, not from invented labels.
"""
import random

from .config import CONTROL_DIMS

GRADE = [(0.8, ["a bit", "slightly", "a little", "somewhat"]), (1.6, ["", "noticeably", "clearly"]),
         (2.6, ["much", "a lot", "far"])]

DIM_PHRASES = {  # dim: (phrases when z>0, phrases when z<0); '{g}' = grade word
    "rate": (["{g} faster", "{g} more quickly", "at a {g} quicker pace", "{g} hurried"],
             ["{g} slower", "{g} more slowly", "at a {g} slower pace", "{g} unhurried", "{g} more deliberately"]),
    "loud": (["{g} louder", "{g} more loudly", "with a {g} raised voice", "{g} more forcefully", "projecting {g} more"],
             ["{g} quieter", "{g} more softly", "{g} hushed", "in a {g} quieter voice", "{g} more gently"]),
    "f0_med": (["{g} higher-pitched", "in a {g} higher voice", "with a {g} higher pitch"],
               ["{g} lower-pitched", "in a {g} deeper voice", "with a {g} lower pitch"]),
    "f0_iqr": (["with {g} more expressive intonation", "{g} more animated", "with {g} livelier pitch movement",
                "{g} more sing-song"],
               ["{g} more monotone", "{g} flatter, with little intonation", "{g} more level and even", "{g} deadpan"]),
    "f0_end": (["with a {g} rising ending", "ending on an upward inflection", "rising at the end like a question"],
               ["with a {g} falling ending", "ending on a firm downward inflection", "dropping at the end"]),
    "pause": (["with {g} more pauses", "{g} haltingly", "pausing {g} more between phrases"],
              ["without pausing", "{g} more fluidly, in one breath", "{g} more continuously"]),
    "tilt": (["with a {g} brighter, sharper tone", "{g} more edgy", "with a {g} more forceful timbre"],
             ["with a {g} softer, mellower tone", "{g} more muted", "with a {g} darker tone"]),
    "voiced": (["with a {g} fuller voice", "{g} more fully voiced"],
               ["in a whisper", "whispering", "{g} breathy, almost whispered", "under your breath"]),
    "flat": (["{g} breathier", "{g} airier"], ["with a {g} clearer tone", "{g} crisper"]),
    "centroid": (["{g} brighter", "with {g} crisper articulation"], ["{g} more muffled", "{g} duller"]),
    "cpps": (["with a {g} clearer, steadier voice", "{g} more resonant"],
             ["with a {g} shakier voice", "{g} rougher", "{g} unsteady", "with a {g} strained voice"]),
    "h1h2": (["{g} breathier and softer", "with a {g} soft, breathy quality"], ["{g} more pressed", "{g} tenser"]),
}

TAG_PHRASES = {
    "neutral": ["normally", "in a neutral tone", "plainly", "in a calm, matter-of-fact way", "like reading the news",
                "without any particular emotion", "in your regular voice"],
    "anger": ["angrily", "furiously", "like you're really mad at someone", "with irritation", "in a fit of rage",
              "as if you've been wronged", "annoyed and heated"],
    "sadness": ["sadly", "as if something bad just happened", "heartbroken", "with sorrow", "gloomily",
                "like you're about to cry", "in a dejected way"],
    "happiness": ["happily", "cheerfully", "with joy", "in a bright, upbeat way", "like you just got great news"],
    "fear": ["fearfully", "as if you're terrified", "nervously, afraid", "with panic in your voice", "scared"],
    "disgust": ["with disgust", "as if something smells awful", "repulsed", "grossed out"],
    "confusion": ["confused", "puzzled", "as if you don't understand what's going on", "bewildered"],
    "amazement": ["amazed", "in awe", "astonished", "as if you can't believe your eyes"],
    "amusement": ["amused", "as if you find it funny", "with a laugh in your voice", "playfully"],
    "adoration": ["adoringly", "lovingly", "warmly, full of affection", "tenderly"],
    "contentment": ["contentedly", "satisfied and relaxed", "at ease"],
    "cuteness": ["in a cutesy voice", "like talking to a puppy", "sweetly, baby-talk style"],
    "desire": ["longingly", "with desire", "craving it", "yearning"],
    "disappointment": ["disappointed", "let down", "with disappointment"],
    "distress": ["distressed", "stressed out", "anxious and overwhelmed", "frantically"],
    "ecstasy": ["ecstatically", "overjoyed", "thrilled beyond words", "euphoric"],
    "embarrassment": ["embarrassed", "sheepishly", "awkwardly, a bit ashamed"],
    "guilt": ["guiltily", "remorsefully", "as if you feel guilty"],
    "interest": ["with interest", "curiously", "intrigued"],
    "pain": ["in pain", "as if you're hurting", "wincing"],
    "pride": ["proudly", "full of pride", "triumphantly"],
    "realization": ["as if you just realized something", "with sudden understanding", "aha, realizing"],
    "relief": ["relieved", "with relief", "as if a weight was lifted"],
    "serenity": ["serenely", "peacefully", "calm and tranquil", "soothingly"],
    "whisper": ["whisper", "whispering", "in a whisper", "softly whispered"],
    "loud": ["loudly", "projecting your voice", "as if speaking to a large room"],
    "fast": ["fast", "quickly", "rapidly"],
    "slow": ["slowly", "very slowly and deliberately"],
    "high_pitch": ["in a high-pitched voice", "with a high voice"],
    "low_pitch": ["in a low-pitched voice", "with a deep voice"],
    "laughing": ["while laughing", "laughing as you speak", "giggling"],
    "enunciated": ["enunciating every word", "very clearly articulated", "with careful enunciation"],
    "emphasis": ["with emphasis", "stressing the key words"],
}

FRAMES_IMP = ["Speak {x}.", "Say this {x}.", "Read it {x}.", "[{x}]", "[speak {x}]", "{x}", "Deliver the line {x}.",
              "Please talk {x}.", "Can you say it {x}?", "Voice: {x}."]
FRAMES_DESC = ["A voice speaking {x}.", "The speaker talks {x}.", "Delivery: {x}.", "Someone saying this {x}."]


def grade_word(az):
    word = ""
    for th, words in GRADE:
        if az >= th:
            word = random.choice(words)
    return word


def dim_phrase(dim, z):
    pos, neg = DIM_PHRASES[dim]
    ph = random.choice(pos if z > 0 else neg)
    ph = " ".join(ph.replace("{g}", grade_word(abs(z))).split())
    for a, b in (("a a ", "a "), ("a a bit", "a bit"), ("a far ", "a far "), ("a much ", "a much "),
                 ("at a a lot", "at a much"), ("a a lot", "a much"), ("with a far", "with a far")):
        ph = ph.replace(a, b)
    return ph


def caption(tag, z, salient=0.8, max_dims=2, p_tag=0.7):
    """Returns (text, used_tag or 'none', list of dims described)."""
    parts, used_dims, used_tag = [], [], "none"
    if tag in TAG_PHRASES and tag != "none" and random.random() < p_tag:
        parts.append(random.choice(TAG_PHRASES[tag]))
        used_tag = tag
    cand = [(d, z[i]) for i, d in enumerate(CONTROL_DIMS) if z[i] == z[i] and abs(z[i]) >= salient]
    random.shuffle(cand)
    k = random.randint(0 if parts else 1, max_dims)
    for d, v in cand[:k]:
        parts.append(dim_phrase(d, v)); used_dims.append(d)
    if not parts:
        parts.append(random.choice(TAG_PHRASES["neutral"])); used_tag = "neutral"
    x = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    frame = random.choice(FRAMES_IMP if random.random() < 0.75 else FRAMES_DESC)
    return frame.replace("{x}", x), used_tag, used_dims
