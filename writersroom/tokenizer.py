"""
Module 2: Tokenization & The Colloquial Speech Problem
======================================================
ColloquialTokenizer addressing conversational elisions, sitcom slang,
slang expansion, subword BPE simulation, and OOV (Out-Of-Vocabulary) analysis.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Dict, List, Set, Tuple


class ColloquialTokenizer:
    """
    Tokenizer specialized for conversational and comedic dialogue.
    
    Handles:
    1. Conversational elisions & slang expansion ('gonna' -> 'going to')
    2. Sitcom catchphrase preservation ('Bazinga', 'Pivot')
    3. Punctuation stripping / normalization
    4. Out-of-Vocabulary (OOV) simulation across training vs test splits
    """

    DEFAULT_CONTRACTIONS: dict[str, str] = {
        r"\bgonna\b": "going to",
        r"\bwanna\b": "want to",
        r"\bgotta\b": "got to",
        r"\bdunno\b": "do not know",
        r"\bc'mon\b": "come on",
        r"\bain't\b": "is not",
        r"\by'all\b": "you all",
        r"\bhowdy\b": "how do you do",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bshan't\b": "shall not",
        r"\bi'm\b": "i am",
        r"\byou're\b": "you are",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
        r"\bit's\b": "it is",
        r"\bwe're\b": "we are",
        r"\bthey're\b": "they are",
        r"\bi've\b": "i have",
        r"\byou've\b": "you have",
        r"\bwe've\b": "we have",
        r"\bthey've\b": "they have",
        r"\bi'll\b": "i will",
        r"\byou'll\b": "you will",
        r"\bhe'll\b": "he will",
        r"\bshe'll\b": "she will",
        r"\bwe'll\b": "we will",
        r"\bthey'll\b": "they will",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bshouldn't\b": "should not",
        r"\bcouldn't\b": "could not",
        r"\bwouldn't\b": "would not",
        r"\bhow you doin'\b": "how are you doing",
        r"\bhow you doin\b": "how are you doing",
    }

    PUNCTUATION_REGEX = re.compile(r"[^\w\s\']")
    WORD_REGEX = re.compile(r"\b[A-Za-z0-9]+(?:'[A-Za-z]+)?\b")

    def __init__(self, custom_slang_map: dict[str, str] | None = None):
        self.slang_map = dict(self.DEFAULT_CONTRACTIONS)
        if custom_slang_map:
            self.slang_map.update(custom_slang_map)

    def expand_contractions(self, text: str) -> str:
        """Expand colloquial elisions and conversational contractions."""
        expanded = text.lower()
        for pattern, replacement in self.slang_map.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        return expanded

    def tokenize_basic(self, text: str, lowercase: bool = True) -> list[str]:
        """Simple whitespace / alphanumeric tokenizer without slang expansion."""
        target = text.lower() if lowercase else text
        return self.WORD_REGEX.findall(target)

    def tokenize_colloquial(
        self,
        text: str,
        expand_slang: bool = True,
        lowercase: bool = True
    ) -> list[str]:
        """
        Tokenize dialogue with slang expansion and punctuation normalization.
        """
        processed = text.lower() if lowercase else text
        if expand_slang:
            processed = self.expand_contractions(processed)
        return self.WORD_REGEX.findall(processed)

    def simulate_bpe_subwords(
        self,
        word: str,
        known_subwords: set[str] | None = None
    ) -> list[str]:
        """
        Simulate Byte-Pair Encoding subword segmentation.
        
        Shows students how an OOV word like 'Bazinga' decomposes into:
        ['Ba', '##zin', '##ga'] rather than being rejected as <UNK>.
        """
        if not word:
            return []
        
        # Educational subword dictionary for sitcom terms
        common_morphemes = known_subwords or {
            "ba", "zin", "ga", "piv", "ot", "paleo", "nto", "logy", "raptor",
            "un", "pre", "dict", "able", "ing", "ed", "s", "er", "est", "ly",
            "multi", "verse", "hydro", "dyna", "mics"
        }

        w = word.lower()
        pieces: list[str] = []
        i = 0
        while i < len(w):
            matched = False
            for step in range(len(w), i, -1):
                chunk = w[i:step]
                if chunk in common_morphemes or step - i <= 2:
                    tag = chunk if i == 0 else f"##{chunk}"
                    pieces.append(tag)
                    i = step
                    matched = True
                    break
            if not matched:
                tag = w[i] if i == 0 else f"##{w[i]}"
                pieces.append(tag)
                i += 1
        return pieces

    def analyze_oov(
        self,
        train_tokens: list[str],
        test_tokens: list[str],
        vocab_cutoff: int = 1
    ) -> dict[str, any]:
        """
        Measure Out-Of-Vocabulary (OOV) rate when test dialogue is checked
        against a vocabulary constructed from training transcripts.
        """
        counts = Counter(train_tokens)
        vocab = {w for w, count in counts.items() if count >= vocab_cutoff}
        
        total_test = len(test_tokens)
        if total_test == 0:
            return {"vocab_size": len(vocab), "oov_count": 0, "oov_rate": 0.0, "oov_tokens": []}

        oov_list = [t for t in test_tokens if t not in vocab]
        oov_rate = len(oov_list) / total_test

        return {
            "vocab_size": len(vocab),
            "total_test_tokens": total_test,
            "oov_count": len(oov_list),
            "oov_rate": round(oov_rate, 4),
            "oov_tokens": sorted(list(set(oov_list))),
        }
