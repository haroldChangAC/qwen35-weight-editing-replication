"""Stand-in for the authors' unpublished mcqa.loop_signals, reimplemented from the rule stated in
mathqa.loop_signals' docstring: in the last 200 words, the 4-word positions whose phrase occurs 3+ times
there are at least half of all positions. Used only for loop diagnostics, never for grading."""
from collections import Counter
def loop_signals(text):
    w = text.split()[-200:]
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    if not grams:
        return {"repetitive_end": False}
    c = Counter(grams)
    return {"repetitive_end": sum(c[g] >= 3 for g in grams) >= len(grams) / 2}
