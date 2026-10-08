"""Text -> ARPAbet phoneme ids with word indices (g2p_en)."""
import re
from functools import lru_cache

_PUNCT = list(",.?!;:-")
_ARPA = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
         "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
SYMBOLS = ["<pad>", "<sp>"] + _PUNCT + [p + s for p in _ARPA for s in (["0", "1", "2"] if p in _VOWELS else [""])]
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
PAD, SP = 0, 1


@lru_cache(maxsize=1)
def _g2p():
    from g2p_en import G2p
    return G2p()


def normalize(text: str) -> str:
    text = text.replace("—", ", ").replace("–", ", ").replace("…", ". ")
    text = re.sub(r"[\"“”‘’`]", lambda m: "'" if m.group(0) in "‘’`" else "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_phonemes(text: str):
    """Returns (ids, word_idx, words). Word boundaries become <sp> tokens with word_idx -1;
    punctuation tokens get word_idx -1 as well. A trailing <sp> pads the end."""
    text = normalize(text)
    raw = _g2p()(text)
    ids, widx, words = [SP], [-1], []
    cur, wi = [], -1
    for tok in raw + [" "]:
        if tok == " ":
            if cur:
                wi += 1
                ids += [SYM2ID[p] for p in cur]
                widx += [wi] * len(cur)
                cur = []
                ids.append(SP); widx.append(-1)
            continue
        if tok in SYM2ID and tok not in ("<pad>", "<sp>"):
            if tok in _PUNCT:
                if cur:
                    wi += 1
                    ids += [SYM2ID[p] for p in cur]
                    widx += [wi] * len(cur)
                    cur = []
                ids.append(SYM2ID[tok]); widx.append(-1)
            else:
                cur.append(tok)
    # words list re-derived from the normalised text for display (best effort alignment)
    words = re.findall(r"[A-Za-z0-9']+", text)
    return ids, widx, words
