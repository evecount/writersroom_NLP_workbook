"""
Module 6: Vector Semantics & Character Tropes (Embeddings)
==========================================================
SitcomEmbeddingEngine handling Word2Vec training on sitcom scripts,
vector arithmetic analogies (Ross - Paleontology + Physics ≈ Sheldon),
and personality radar scoring via cosine similarity against semantic anchors.
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SitcomEmbeddingEngine:
    """
    Vector semantics engine for sitcom dialogue corpora.
    
    Provides:
    - Word2Vec training on tokenized sentences
    - Vector arithmetic: v(Ross) - v(paleontology) + v(physics) ~ v(Sheldon)
    - Character personality projection against affective semantic anchors:
      [neurotic, romantic, cynical, naive, academic]
    - Cosine similarity computation
    """

    ANCHOR_TRAITS = {
        "neurotic": ["anxious", "nervous", "worry", "panic", "yell", "stress", "catastrophe", "terrible", "problem"],
        "romantic": ["love", "marriage", "date", "relationship", "affection", "heart", "marry", "sweetie", "kiss"],
        "cynical": ["sarcastic", "joke", "ridiculous", "cynic", "torture", "mock", "useless", "shut", "kill"],
        "academic": ["physics", "science", "equation", "paleontology", "theory", "quantum", "calculation", "fossil"],
        "naive": ["pizza", "sandwich", "food", "simple", "curb", "kidney", "cool", "fun", "friend"],
    }

    def __init__(self, vector_size: int = 50, window: int = 5, min_count: int = 1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.vocab: dict[str, int] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.gensim_model = None

    def train(self, sentences: list[list[str]]) -> None:
        """
        Train Word2Vec embeddings on sitcom sentences using gensim,
        with a deterministic SVD/PPMI fallback if gensim is unavailable.
        """
        try:
            from gensim.models import Word2Vec
            self.gensim_model = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=1,  # Skip-gram
                epochs=30,
                seed=42,
            )
            for word in self.gensim_model.wv.index_to_key:
                self.vectors[word.lower()] = self.gensim_model.wv[word]
            self.vocab = {w: i for i, w in enumerate(self.vectors.keys())}
        except Exception:
            # Deterministic PPMI + co-occurrence embedding fallback
            self._train_cooccurrence_embeddings(sentences)

    def _train_cooccurrence_embeddings(self, sentences: list[list[str]]) -> None:
        """Co-occurrence PPMI fallback when gensim is not installed."""
        cooccur = defaultdict(lambda: defaultdict(int))
        unigrams = defaultdict(int)

        for sent in sentences:
            tokens = [t.lower() for t in sent if t.isalnum()]
            for i, target in enumerate(tokens):
                unigrams[target] += 1
                start = max(0, i - self.window)
                end = min(len(tokens), i + self.window + 1)
                for j in range(start, end):
                    if i != j:
                        context = tokens[j]
                        cooccur[target][context] += 1

        all_words = [w for w, c in unigrams.items() if c >= self.min_count]
        self.vocab = {w: i for i, w in enumerate(all_words)}
        dim = min(self.vector_size, len(all_words))
        np.random.seed(42)

        for w in all_words:
            # Deterministic pseudo-dense representation based on context distribution
            vec = np.zeros(dim, dtype=np.float32)
            for ctx, count in cooccur[w].items():
                if ctx in self.vocab:
                    idx = self.vocab[ctx] % dim
                    vec[idx] += count
            norm = np.linalg.norm(vec)
            self.vectors[w] = vec / norm if norm > 0 else np.ones(dim, dtype=np.float32) / math.sqrt(dim)

    def get_vector(self, word: str) -> np.ndarray | None:
        """Retrieve word vector, returning None if OOV."""
        return self.vectors.get(word.lower())

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two dense vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def word_similarity(self, word_a: str, word_b: str) -> float:
        """Calculate cosine similarity between two words."""
        va = self.get_vector(word_a)
        vb = self.get_vector(word_b)
        if va is None or vb is None:
            return 0.0
        return self.cosine_similarity(va, vb)

    def vector_analogy(
        self,
        positive: list[str],
        negative: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Perform vector arithmetic:
        target = sum(positives) - sum(negatives)
        Find nearest neighbor words.
        """
        if not self.vectors:
            return []

        sample_vec = next(iter(self.vectors.values()))
        target_vec = np.zeros_like(sample_vec)

        query_set = {w.lower() for w in positive + negative}
        for pos in positive:
            vec = self.get_vector(pos)
            if vec is not None:
                target_vec += vec
        for neg in negative:
            vec = self.get_vector(neg)
            if vec is not None:
                target_vec -= vec

        scores = []
        for word, vec in self.vectors.items():
            if word in query_set:
                continue
            sim = self.cosine_similarity(target_vec, vec)
            scores.append((word, round(sim, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def build_character_vector(self, dialogue_tokens: list[str]) -> np.ndarray:
        """Build character centroid vector by averaging their dialogue word vectors."""
        valid_vectors = [
            self.vectors[w.lower()]
            for w in dialogue_tokens
            if w.lower() in self.vectors
        ]
        if not valid_vectors:
            sample_dim = len(next(iter(self.vectors.values()))) if self.vectors else self.vector_size
            return np.zeros(sample_dim, dtype=np.float32)

        centroid = np.mean(valid_vectors, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def compute_personality_radar(
        self,
        character_vector: np.ndarray
    ) -> dict[str, float]:
        """
        Project character vector onto 5 personality anchor archetypes:
        ['neurotic', 'romantic', 'cynical', 'academic', 'naive'].
        """
        radar_scores = {}
        for trait, seed_words in self.ANCHOR_TRAITS.items():
            trait_vectors = [
                self.vectors[w] for w in seed_words if w in self.vectors
            ]
            if not trait_vectors:
                radar_scores[trait] = 0.0
                continue
            trait_centroid = np.mean(trait_vectors, axis=0)
            sim = max(0.0, self.cosine_similarity(character_vector, trait_centroid))
            radar_scores[trait] = round(sim, 3)

        return radar_scores
