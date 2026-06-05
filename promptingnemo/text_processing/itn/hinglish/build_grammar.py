#!/usr/bin/env python3
"""Build Hinglish ITN .far using compositional WFSTs.

Same approach as NeMo's English CardinalFst — compose small FSTs for
digits/ties/teens/compounds with multiplier FSTs (सौ/हजार/लाख/करोड़),
instead of enumerating every combination as a flat string map.

Result: ~3 MB .far (vs 26 MB from enumeration) with identical coverage.

Usage:
    python build_grammar.py                    # Build and test
    python build_grammar.py --output itn.far   # Custom output path
    python build_grammar.py --upload            # Build + upload to GCS

Requires: pynini (pip install pynini)
"""

import argparse
import os
import subprocess
from pathlib import Path

import pynini
from pynini.lib import pynutil

DATA_DIR = Path(__file__).parent

# Shared constants
NEMO_SPACE = pynini.union(*" \t").optimize()
# Safe ASCII chars (exclude brackets which are special in pynini)
_safe_ascii = [chr(i) for i in range(32, 128) if chr(i) not in "[]"]
_devanagari = [chr(i) for i in range(0x0900, 0x0980)]
NEMO_SIGMA = pynini.closure(pynini.union(
    pynini.string_map(_safe_ascii),
    pynini.string_map(_devanagari),
)).optimize()
NEMO_DIGIT = pynini.union(*"0123456789").optimize()
NEMO_ALPHA = pynini.union(
    pynini.string_map([chr(i) for i in range(ord("a"), ord("z") + 1)]),
    pynini.string_map([chr(i) for i in range(ord("A"), ord("Z") + 1)]),
    pynini.string_map(_devanagari),
).optimize()

delete_space = pynutil.delete(NEMO_SPACE)


def _load_tsv(name: str) -> pynini.Fst:
    return pynini.string_file(str(DATA_DIR / name))


class HinglishCardinalFst:
    """Compositional cardinal number FST for English + Hindi + Hinglish."""

    def __init__(self):
        graph_zero = _load_tsv("zero.tsv")
        graph_digit = _load_tsv("digit.tsv")
        graph_ties = _load_tsv("ties.tsv")
        graph_teen = _load_tsv("teen.tsv")
        graph_compound = _load_tsv("compound.tsv")

        # Two-digit: teen | (ties + digit) | compound
        graph_two_digit = pynini.union(
            graph_teen,
            graph_compound,
            graph_ties + delete_space + (graph_digit | pynutil.insert("0")),
        )

        # Hundred: "hundred" | "सौ"
        graph_hundred = pynini.union(
            pynini.cross("hundred", ""),
            pynini.cross("सौ", ""),
            pynini.cross("हंड्रेड", ""),
        )

        # Hundred component: [digit hundred] (teen | ties+digit | compound | 00)
        graph_hundred_component = pynini.union(
            graph_digit + delete_space + graph_hundred,
            pynutil.insert("0"),
        )
        graph_hundred_component += delete_space
        graph_hundred_component += pynini.union(
            graph_teen | graph_compound | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )

        # Filter: at least one non-zero digit
        at_least_one_nonzero = pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        graph_hundred_nonzero = graph_hundred_component @ at_least_one_nonzero

        # Ties component (for Indian system: XX not XXX)
        graph_ties_component = pynini.union(
            graph_teen | graph_compound | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )
        graph_ties_nonzero = graph_ties_component @ at_least_one_nonzero

        # Thousand: "thousand" | "हजार" | "हज़ार" | "थाउज़ेंड"
        delete_thousand = pynini.union(
            pynutil.delete("thousand"),
            pynutil.delete("हजार"),
            pynutil.delete("हज़ार"),
            pynutil.delete("थाउज़ेंड"),
        )

        graph_thousands = pynini.union(
            graph_hundred_nonzero + delete_space + delete_thousand,
            pynutil.insert("000", weight=0.1),
        )

        # International system
        graph_million = pynini.union(
            graph_hundred_nonzero + delete_space + pynutil.delete("million"),
            pynutil.insert("000", weight=0.1),
        )
        graph_billion = pynini.union(
            graph_hundred_nonzero + delete_space + pynutil.delete("billion"),
            pynutil.insert("000", weight=0.1),
        )

        graph_int = (
            graph_billion + delete_space
            + graph_million + delete_space
            + graph_thousands
        )

        # Indian system: lakh, crore
        delete_lakh = pynini.union(
            pynutil.delete("lakh"), pynutil.delete("lakhs"),
            pynutil.delete("लाख"), pynutil.delete("लाखों"),
        )
        delete_crore = pynini.union(
            pynutil.delete("crore"), pynutil.delete("crores"),
            pynutil.delete("करोड़"),
        )

        graph_in_thousands = pynini.union(
            graph_ties_nonzero + delete_space + delete_thousand,
            pynutil.insert("00", weight=0.1),
        )
        graph_in_lakhs = pynini.union(
            graph_ties_nonzero + delete_space + delete_lakh,
            pynutil.insert("00", weight=0.1),
        )
        graph_in_crores = pynini.union(
            graph_ties_nonzero + delete_space + delete_crore,
            pynutil.insert("00", weight=0.1),
        )

        graph_ind = (
            graph_in_crores + delete_space
            + graph_in_lakhs + delete_space
            + graph_in_thousands
        )

        # Full hundreds
        graph_hundreds = pynini.union(
            graph_hundred_component,
            graph_digit + delete_space + graph_hundred + delete_space + pynini.union(
                graph_compound,
                graph_teen,
                (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
                pynutil.insert("00"),
            ),
        )

        graph = pynini.union(
            (graph_int | graph_ind) + delete_space + graph_hundreds,
            graph_zero,
        )

        # Strip leading zeros
        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0")) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT),
            "0",
        )

        # Delete "and" between components
        graph = (
            pynini.cdrewrite(pynutil.delete("and"), NEMO_SPACE, NEMO_SPACE, NEMO_SIGMA)
            @ graph
        ).optimize()

        # Exception: don't convert single words below 13
        try:
            from .utils import num_to_word
        except ImportError:
            from utils import num_to_word
        labels_exception = [num_to_word(x) for x in range(0, 13)]
        graph_exception = pynini.union(*labels_exception).optimize()

        self.graph_no_exception = graph
        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        # Tag as cardinal
        final = (
            pynutil.insert('integer: "')
            + self.graph
            + pynutil.insert('"')
        )
        self.fst = pynutil.insert("cardinal { ") + final + pynutil.insert(" }")


class HinglishMoneyFst:
    """Money FST using cardinal + currency composition."""

    def __init__(self, cardinal: HinglishCardinalFst):
        graph_cardinal = cardinal.graph_no_exception

        unit = _load_tsv("currency.tsv")
        unit_singular = pynini.invert(unit)

        # Plural forms
        unit_plural = unit_singular  # simplified — accept both forms

        graph_unit = pynutil.insert('currency: "') + unit_singular + pynutil.insert('"')

        # "twenty three dollars" -> money { integer_part: "23" currency: "$" }
        graph_integer = (
            pynutil.insert('integer_part: "')
            + graph_cardinal
            + pynutil.insert('"')
        )

        # Cent/paise words
        delete_cent = pynini.union(
            pynutil.delete("cent"), pynutil.delete("cents"),
            pynutil.delete("पैसे"), pynutil.delete("पैसा"), pynutil.delete("पैसों"),
        )

        graph_fractional = (
            pynutil.insert('fractional_part: "')
            + graph_cardinal
            + pynutil.insert('"')
        )

        # Full money: integer + currency [+ "and" + fractional + cent]
        graph = (
            graph_integer + delete_space + graph_unit
            + pynini.closure(
                delete_space + pynutil.delete("and") + delete_space
                + pynutil.insert(" ") + graph_fractional + delete_space + delete_cent,
                0, 1,
            )
        )

        self.fst = pynutil.insert("money { ") + graph + pynutil.insert(" }")


class HinglishPercentFst:
    """Percent FST: "fifty percent" -> "50%" """

    def __init__(self, cardinal: HinglishCardinalFst):
        delete_pct = pynini.union(
            pynutil.delete("percent"),
            pynutil.delete("परसेंट"),
            pynutil.delete("पर्सेंट"),
            pynutil.delete("प्रतिशत"),
        )

        graph = (
            pynutil.insert('decimal { integer_part: "')
            + cardinal.graph_no_exception
            + pynutil.insert('" } units: "%"')
        )
        graph = graph + delete_space + delete_pct

        self.fst = pynutil.insert("measure { ") + graph + pynutil.insert(" }")


class HinglishClassifyFst:
    """Top-level classifier composing all semiotic FSTs."""

    def __init__(self):
        cardinal = HinglishCardinalFst()
        money = HinglishMoneyFst(cardinal)
        percent = HinglishPercentFst(cardinal)

        # Word (passthrough for non-semiotic tokens)
        word_graph = pynutil.insert('name: "') + pynini.closure(
            pynini.union(
                pynini.string_map([chr(i) for i in range(ord("a"), ord("z") + 1)]),
                pynini.string_map([chr(i) for i in range(ord("A"), ord("Z") + 1)]),
                pynini.string_map([chr(i) for i in range(0x0900, 0x0980)]),
                pynini.string_map(list("'-.")),
            ), 1
        ) + pynutil.insert('"')
        word_fst = pynutil.insert("name: ") + word_graph

        # Token wrapper
        token = pynutil.insert("tokens { ") + pynini.union(
            money.fst,
            percent.fst,
            cardinal.fst,
            word_fst,
        ) + pynutil.insert(" }")

        # Full graph: token (space token)*
        graph = token + pynini.closure(delete_space + pynutil.insert(" ") + token)

        self.fst = graph.optimize()


def build(output_path: str = "hinglish_itn.far"):
    """Build and save the Hinglish ITN .far file."""
    print("Building Hinglish compositional ITN grammar...")
    classifier = HinglishClassifyFst()

    far = pynini.Far(output_path, mode="w", arc_type="standard", far_type="default")
    far.add("tokenize_and_classify", classifier.fst)
    far.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    # Test
    test_far = pynini.Far(output_path, mode="r")
    test_far.get_key()
    fst = test_far.get_fst()

    tests = [
        "twenty three dollars",
        "one hundred and forty five",
        "तेईस",
        "पाँच सौ",
        "एक हजार",
        "दो लाख",
        "पैंतालीस",
        "छबीस रुपये",
        "एक हजार दो सौ पैंतालीस रूपए",
        "पचास परसेंट",
        "twenty three dollars and fifty cents",
    ]

    print("\n=== Tests ===")
    for t in tests:
        try:
            lattice = pynini.escape(t) @ fst
            if lattice.start() != pynini.NO_STATE_ID:
                r = pynini.shortestpath(lattice).string()
                print(f"  {t:>45} -> {r[:80]}")
            else:
                print(f"  {t:>45} -> (no match)")
        except Exception as e:
            print(f"  {t:>45} -> ERR: {e}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Hinglish ITN .far grammar")
    parser.add_argument("--output", default="hinglish_itn.far")
    parser.add_argument("--upload", action="store_true", help="Upload to GCS after build")
    args = parser.parse_args()

    far_path = build(args.output)

    if args.upload:
        gcs_path = "gs://whissle-voice-recordings/asr-models/itn_far/en_itn_lower_cased.far"
        subprocess.run(["gsutil", "cp", far_path, gcs_path], check=True)
        print(f"Uploaded to {gcs_path}")
