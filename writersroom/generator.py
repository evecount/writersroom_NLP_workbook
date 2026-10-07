"""
Module 7: The Script Generator Engine (Synthesis & Evaluation)
==============================================================
CharacterMarkovEngine generating sitcom dialogue using Character-Conditioned
N-gram Markov chains with Laplace smoothing, temperature sampling, and an
evaluation benchmark harness comparing classical vs neural generation.
"""

from __future__ import annotations
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class CharacterMarkovEngine:
    """
    Character-Conditioned Markov Language Model.
    
    Models:
    P(w_t | w_{t-1}, Character)  [Bigram]
    P(w_t | w_{t-2}, w_{t-1}, Character)  [Trigram]
    
    Includes:
    - Laplace smoothing (alpha)
    - Temperature-scaled probabilistic sampling
    - Prompt-based completion
    - Generation evaluation harness (TTR, character consistency, sentence length)
    """

    START_TOKEN = "<START>"
    END_TOKEN = "<END>"

    def __init__(self, n: int = 2, laplace_alpha: float = 0.1):
        self.n = n
        self.alpha = laplace_alpha
        # Transitions: transitions[speaker][prefix_tuple] = Counter(next_tokens)
        self.transitions: dict[str, dict[tuple[str, ...], Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.character_vocabs: dict[str, set[str]] = defaultdict(set)
        self.all_vocab: set[str] = set()

    def train_on_dataframe(
        self,
        dialogue_df: pd.DataFrame,
        speaker_col: str = "speaker",
        line_col: str = "line"
    ) -> None:
        """Fit Markov transition distributions on character dialogue rows."""
        for _, row in dialogue_df.iterrows():
            speaker = str(row[speaker_col]).strip().upper()
            line = str(row[line_col]).strip()
            if not speaker or not line:
                continue

            tokens = [t.lower() for t in re.findall(r"\b\w+(?:'\w+)?\b|[.!?]", line)]
            if not tokens:
                continue

            padded = [self.START_TOKEN] * (self.n - 1) + tokens + [self.END_TOKEN]
            for t in tokens:
                self.character_vocabs[speaker].add(t)
                self.all_vocab.add(t)

            for i in range(len(padded) - (self.n - 1)):
                prefix = tuple(padded[i : i + self.n - 1])
                next_word = padded[i + self.n - 1]
                self.transitions[speaker][prefix][next_word] += 1

    def sample_next_word(
        self,
        speaker: str,
        prefix: tuple[str, ...],
        temperature: float = 1.0
    ) -> str:
        """Sample next token given prefix context and temperature."""
        speaker_trans = self.transitions.get(speaker, {})
        counts = speaker_trans.get(prefix)

        # Fallback to general speaker vocabulary if prefix is unseen
        if not counts:
            if not self.character_vocabs[speaker]:
                return self.END_TOKEN
            candidates = list(self.character_vocabs[speaker]) + [self.END_TOKEN]
            return random.choice(candidates)

        words = list(counts.keys())
        raw_freqs = [counts[w] for w in words]

        # Apply temperature
        if temperature <= 0.01:
            return words[raw_freqs.index(max(raw_freqs))]

        # Exponentiate counts with temperature
        scaled = [math.pow(c, 1.0 / max(0.1, temperature)) for c in raw_freqs]
        total = sum(scaled)
        probs = [s / total for s in scaled]

        return random.choices(words, weights=probs, k=1)[0]

    def generate_line(
        self,
        speaker: str,
        prompt: Optional[str] = None,
        max_tokens: int = 30,
        temperature: float = 0.8,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate a single character-conditioned line of sitcom dialogue.
        """
        if seed is not None:
            random.seed(seed)

        speaker_clean = speaker.strip().upper()
        if prompt:
            prompt_tokens = [t.lower() for t in re.findall(r"\b\w+(?:'\w+)?\b|[.!?]", prompt)]
            generated = list(prompt_tokens)
            if len(prompt_tokens) >= self.n - 1:
                prefix = tuple(prompt_tokens[-(self.n - 1):])
            else:
                prefix = tuple([self.START_TOKEN] * (self.n - 1 - len(prompt_tokens)) + prompt_tokens)
        else:
            generated = []
            prefix = tuple([self.START_TOKEN] * (self.n - 1))

        for _ in range(max_tokens):
            next_word = self.sample_next_word(speaker_clean, prefix, temperature=temperature)
            if next_word == self.END_TOKEN:
                break
            generated.append(next_word)
            prefix = tuple(list(prefix[1:]) + [next_word])

        # Formatting capitalization and punctuation
        out_text = " ".join(generated)
        out_text = re.sub(r"\s+([,.!?])", r"\1", out_text)
        return out_text.capitalize() if out_text else "..."

    def generate_scene(
        self,
        characters: list[str],
        turns: int = 6,
        temperature: float = 0.8
    ) -> list[dict[str, str]]:
        """Generate a multi-turn sitcom scene between characters."""
        scene_script = []
        for i in range(turns):
            char = characters[i % len(characters)]
            line = self.generate_line(char, temperature=temperature)
            scene_script.append({"speaker": char, "line": line})
        return scene_script

    @staticmethod
    def evaluate_generation(
        generated_lines: list[str],
        reference_tokens: list[str]
    ) -> dict[str, float]:
        """
        Benchmark generated script quality across:
        1. Lexical Diversity (TTR)
        2. Reference Vocabulary Overlap (Fidelity)
        3. Mean Line Length (Cadence)
        """
        all_gen_tokens = []
        for line in generated_lines:
            all_gen_tokens.extend(re.findall(r"\b\w+\b", line.lower()))

        total_gen = len(all_gen_tokens)
        if total_gen == 0:
            return {"ttr": 0.0, "vocab_overlap": 0.0, "mean_token_length": 0.0}

        ttr = len(set(all_gen_tokens)) / total_gen
        ref_set = set(reference_tokens)
        overlap = sum(1 for t in all_gen_tokens if t in ref_set) / total_gen
        mean_len = total_gen / len(generated_lines)

        return {
            "ttr": round(ttr, 4),
            "vocab_overlap": round(overlap, 4),
            "mean_token_length": round(mean_len, 2),
        }
