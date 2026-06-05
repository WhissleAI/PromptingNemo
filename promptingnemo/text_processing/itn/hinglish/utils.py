"""Utilities for Hinglish ITN grammar building."""

_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve",
}


def num_to_word(n: int) -> str:
    return _ONES.get(n, str(n))
