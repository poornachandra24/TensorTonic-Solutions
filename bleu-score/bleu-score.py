from collections import Counter
import math

def bleu_score(candidate, reference, max_n):
    # Edge case
    if not candidate:
        return 0.0

    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    precisions = []

    for n in range(1, max_n + 1):
        cand_ngrams = Counter(get_ngrams(candidate, n))
        ref_ngrams = Counter(get_ngrams(reference, n))

        overlap = 0
        for ng, count in cand_ngrams.items():
            overlap += min(count, ref_ngrams.get(ng, 0))

        total = sum(cand_ngrams.values())

        if total == 0:
            return 0.0  # required by spec

        p_n = overlap / total
        if p_n == 0:
            return 0.0  # required: if any precision is zero → return 0

        precisions.append(p_n)

    # Geometric mean (uniform weights)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    # Brevity penalty
    c_len = len(candidate)
    r_len = len(reference)

    bp = math.exp(min(0, 1 - r_len / c_len))

    return bp * geo_mean