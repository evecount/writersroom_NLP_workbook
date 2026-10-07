"""
Module 3: Lexical Diversity & Vocabulary Saturation
===================================================
LexicalAnalyzer calculating Types vs Tokens, Type-Token Ratio (TTR),
Root TTR, Hapax Legomena extraction, Cumulative Vocabulary Growth, and Zipf's Law.
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Any, Dict, List, Tuple


class LexicalAnalyzer:
    """
    Quantitative linguistic analysis for sitcom character vocabularies.
    
    Provides:
    - Standard TTR (V / N)
    - Root TTR (V / sqrt(N))
    - Hapax Legomena (words occurring exactly once)
    - Cumulative vocabulary saturation tracking
    - Zipf's Law log-log regression fitting
    """

    @staticmethod
    def count_tokens_and_types(tokens: list[str]) -> tuple[int, int]:
        """Return total tokens (N) and unique types (V)."""
        return len(tokens), len(set(tokens))

    @staticmethod
    def compute_ttr(tokens: list[str]) -> float:
        """
        Calculate standard Type-Token Ratio: TTR = V / N.
        Ranges from 0.0 to 1.0 (higher = richer vocabulary).
        """
        n = len(tokens)
        if n == 0:
            return 0.0
        v = len(set(tokens))
        return round(v / n, 4)

    @staticmethod
    def compute_root_ttr(tokens: list[str]) -> float:
        """
        Calculate Root Type-Token Ratio: Root TTR = V / sqrt(N).
        Mitigates token-length bias when comparing characters with
        differing amounts of screen time / total dialogue lines.
        """
        n = len(tokens)
        if n == 0:
            return 0.0
        v = len(set(tokens))
        return round(v / math.sqrt(n), 4)

    @staticmethod
    def find_hapaxes(tokens: list[str]) -> list[str]:
        """
        Extract Hapax Legomena: words that appear exactly once in the corpus.
        Reflects characters who use rare, specialized, or academic vocabulary
        (e.g. Sheldon Cooper's idiosyncratic academic jargon).
        """
        counts = Counter(tokens)
        return sorted([w for w, c in counts.items() if c == 1])

    @staticmethod
    def compute_cumulative_growth(tokens: list[str]) -> tuple[list[int], list[int]]:
        """
        Track cumulative unique vocabulary types as a function of token progress.
        Returns (token_steps, cumulative_types).
        """
        seen_types = set()
        token_steps = []
        growth = []

        for i, token in enumerate(tokens, start=1):
            seen_types.add(token)
            if i % 10 == 0 or i == len(tokens):
                token_steps.append(i)
                growth.append(len(seen_types))

        return token_steps, growth

    @staticmethod
    def fit_zipf_law(tokens: list[str]) -> dict[str, any]:
        """
        Test Zipf's law: f(r) ~ C / r^s  <=>  log(f) = log(C) - s * log(r).
        Performs least-squares linear regression on log(rank) vs log(frequency).
        """
        if not tokens:
            return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "ranks": [], "freqs": []}

        counts = Counter(tokens).most_common()
        ranks = list(range(1, len(counts) + 1))
        freqs = [c[1] for c in counts]
        words = [c[0] for c in counts]

        # Log-transformed variables
        log_r = [math.log(r) for r in ranks]
        log_f = [math.log(f) for f in freqs]

        n = len(ranks)
        if n < 2:
            return {
                "slope": -1.0,
                "intercept": 0.0,
                "r_squared": 1.0,
                "ranks": ranks,
                "freqs": freqs,
                "words": words,
            }

        mean_x = sum(log_r) / n
        mean_y = sum(log_f) / n

        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_r, log_f))
        den = sum((x - mean_x) ** 2 for x in log_r)

        slope = num / den if den != 0 else -1.0
        intercept = mean_y - slope * mean_x

        # R-squared
        ss_tot = sum((y - mean_y) ** 2 for y in log_f)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_r, log_f))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return {
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r_squared": round(r2, 4),
            "ranks": ranks,
            "freqs": freqs,
            "words": words,
        }
