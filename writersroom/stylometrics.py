"""
Module 4: Stylometric Fingerprinting via Conditional Frequencies
=================================================================
StylometricFingerprinter calculating Conditional Frequency Distributions (CFD),
characteristic discourse marker matrices, PMI collocation discovery, and dispersion.
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
import pandas as pd


class StylometricFingerprinter:
    """
    Stylometric character profiling tool.
    
    Provides:
    - Character-conditioned word frequency distributions (CFDs)
    - Discourse marker / modal verb usage rates per 1,000 words
    - Pointwise Mutual Information (PMI) collocation extraction for catchphrases
    - Lexical dispersion indexing across script narratives
    """

    DEFAULT_DISCOURSE_MARKERS = [
        "literally",
        "honestly",
        "technically",
        "suppose",
        "actually",
        "obviously",
        "bazinga",
        "pivot",
        "break",
        "giddyup",
        "neurotic",
        "could",
    ]

    @staticmethod
    def build_character_cfd(
        dialogue_df: pd.DataFrame,
        words_of_interest: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Build a 2D matrix where Rows = Character, Columns = Target Words,
        and values represent raw frequency counts.
        """
        import re
        targets = [w.lower() for w in (words_of_interest or StylometricFingerprinter.DEFAULT_DISCOURSE_MARKERS)]
        cfd: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_words: dict[str, int] = defaultdict(int)

        for _, row in dialogue_df.iterrows():
            speaker = str(row.get("speaker", "UNKNOWN")).strip().upper()
            line = str(row.get("line", "")).lower()
            tokens = re.findall(r"\b\w+\b", line)
            total_words[speaker] += len(tokens)
            # Ensure speaker exists in cfd
            _ = cfd[speaker]

            for w in tokens:
                if w in targets:
                    cfd[speaker][w] += 1

        df_matrix = pd.DataFrame.from_dict(cfd, orient="index").fillna(0).astype(int)
        for t in targets:
            if t not in df_matrix.columns:
                df_matrix[t] = 0

        df_matrix["TOTAL_TOKENS"] = [total_words[s] for s in df_matrix.index]
        return df_matrix.sort_index()

    @staticmethod
    def compute_normalized_marker_rates(
        cfd_matrix: pd.DataFrame,
        per_k_tokens: float = 1000.0
    ) -> pd.DataFrame:
        """
        Normalize counts to frequency per 1,000 tokens so characters with
        differing line volumes can be fairly compared.
        """
        norm_df = cfd_matrix.copy()
        totals = norm_df["TOTAL_TOKENS"].replace(0, 1)

        marker_cols = [c for c in norm_df.columns if c != "TOTAL_TOKENS"]
        for col in marker_cols:
            norm_df[col] = (norm_df[col] / totals) * per_k_tokens
            norm_df[col] = norm_df[col].round(2)

        return norm_df

    @staticmethod
    def extract_bigram_collocations_pmi(
        tokens: list[str],
        min_freq: int = 2,
        top_n: int = 15
    ) -> list[dict[str, any]]:
        """
        Compute Pointwise Mutual Information (PMI) for bigrams:
            PMI(w1, w2) = log2( P(w1, w2) / (P(w1) * P(w2)) )
        
        Extracts multi-word collocations and catchphrases like
        ('on', 'a'), ('a', 'break'), ('how', 'you'), ('you', 'doin')
        without hardcoded keyword rules.
        """
        if len(tokens) < 2:
            return []

        clean_tokens = [t.lower() for t in tokens if t.isalnum()]
        total_tokens = len(clean_tokens)
        unigram_counts = Counter(clean_tokens)

        bigrams = list(zip(clean_tokens[:-1], clean_tokens[1:]))
        total_bigrams = len(bigrams)
        bigram_counts = Counter(bigrams)

        collocations = []
        for (w1, w2), count in bigram_counts.items():
            if count < min_freq:
                continue

            p_w1_w2 = count / total_bigrams
            p_w1 = unigram_counts[w1] / total_tokens
            p_w2 = unigram_counts[w2] / total_tokens

            pmi = math.log2(p_w1_w2 / (p_w1 * p_w2))
            collocations.append({
                "bigram": f"{w1} {w2}",
                "w1": w1,
                "w2": w2,
                "count": count,
                "pmi": round(pmi, 3),
            })

        collocations.sort(key=lambda x: (x["count"], x["pmi"]), reverse=True)
        return collocations[:top_n]

    @staticmethod
    def compute_lexical_dispersion(
        tokens: list[str],
        target_words: list[str]
    ) -> dict[str, list[int]]:
        """
        Track position indices of target words across chronological dialogue tokens.
        Enables plotting dispersion plots across season episodes.
        """
        targets = {w.lower() for w in target_words}
        positions: dict[str, list[int]] = {w: [] for w in targets}

        for idx, token in enumerate(tokens):
            w = token.lower()
            if w in targets:
                positions[w].append(idx)

        return positions
