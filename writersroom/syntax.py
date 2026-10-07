"""
Module 5: Syntactic Structure & Rule-Based Matchers (spaCy)
===========================================================
SitcomSyntaxParser utilizing Part-of-Speech (POS) tags, dependency trees,
and pattern matchers to detect comedic cadences (Chandler's rhetorical pattern)
and map character habits via verb-direct object (dobj) extraction.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple


class SitcomSyntaxParser:
    """
    Syntactic analyzer for sitcom comedic structures.
    
    Provides:
    - Chandler Bing's rhetorical cadence matcher: [Could] + [PRON] + [BE] + [ADV]* + [ADJ]
    - Sarcastic interrogative patterns
    - Dependency parse traversal to link Action Verbs to Direct Objects (dobj)
    - Fallback heuristic parser if spaCy model is not pre-installed
    """

    CHANDLER_CADENCE_REGEX = re.compile(
        r"\bcould\s+(?:[a-zA-Z0-9\'\-]+(?:\s+[a-zA-Z0-9\'\-]+)?)\s+(?:be|BE)\s+(?:any\s+)?([a-z]+)\b",
        re.IGNORECASE
    )

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.model_name = spacy_model
        self.nlp = None
        self.matcher = None
        self._init_spacy()

    def _init_spacy(self):
        """Attempt to load spaCy model and configure Matcher rules."""
        try:
            import spacy
            from spacy.matcher import Matcher
            self.nlp = spacy.load(self.model_name)
            self.matcher = Matcher(self.nlp.vocab)

            # Define Chandler's Rhetorical Cadence:
            # e.g., "Could this BE any more depressing?" or "Could this coffee BE any hotter?"
            chandler_pattern = [
                {"LOWER": "could"},
                {"POS": {"IN": ["PRON", "DET", "NOUN", "PROPN"]}, "OP": "+"},
                {"LEMMA": "be"},
                {"POS": "ADV", "OP": "*"},
                {"POS": "ADJ"}
            ]
            self.matcher.add("CHANDLER_RHETORICAL", [chandler_pattern])
            
            # Sarcastic Negation Pattern: "Why don't you..." / "Why wouldn't..."
            sarcastic_pattern = [
                {"LOWER": {"IN": ["why", "how", "what"]}},
                {"LEMMA": {"IN": ["do", "would", "could", "should"]}},
                {"LOWER": "not", "OP": "?"},
                {"POS": "PRON"},
                {"POS": "ADV", "OP": "*"},
                {"POS": "VERB"}
            ]
            self.matcher.add("SARCASTIC_INTERROGATIVE", [sarcastic_pattern])
        except Exception:
            # Graceful fallback when running in lightweight or offline environments
            self.nlp = None
            self.matcher = None

    def find_rhetorical_cadences(self, text: str) -> list[dict[str, any]]:
        """
        Detect Chandler's signature question pattern:
        'Could [subject] [BE] any [adjective]?'
        """
        results = []
        if self.nlp and self.matcher:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                string_id = self.nlp.vocab.strings[match_id]
                span = doc[start:end]
                results.append({
                    "pattern": string_id,
                    "matched_text": span.text,
                    "tokens": [t.text for t in span],
                    "pos_tags": [t.pos_ for t in span],
                })
        else:
            # Fallback regex extraction
            for match in self.CHANDLER_CADENCE_REGEX.finditer(text):
                results.append({
                    "pattern": "CHANDLER_RHETORICAL",
                    "matched_text": match.group(0),
                    "tokens": match.group(0).split(),
                    "pos_tags": ["AUX", "PRON", "VERB", "ADV", "ADJ"][:len(match.group(0).split())],
                })
        return results

    def extract_verb_direct_objects(
        self,
        text: str,
        target_verbs: list[str] | None = None
    ) -> list[dict[str, str]]:
        """
        Extract (Verb -> Direct Object) pairs from sentence dependency trees.
        
        Example:
        'Joey eats pepperoni pizza' -> {'verb': 'eat', 'dobj': 'pizza'}
        'Ross assembles Scandinavian furniture' -> {'verb': 'assemble', 'dobj': 'furniture'}
        """
        pairs = []
        target_lemmas = {v.lower() for v in target_verbs} if target_verbs else None

        if self.nlp:
            doc = self.nlp(text)
            for token in doc:
                if token.pos_ == "VERB":
                    lemma = token.lemma_.lower()
                    if target_lemmas and lemma not in target_lemmas:
                        continue
                    # Traverse children to find direct object
                    for child in token.children:
                        if child.dep_ in ("dobj", "obj"):
                            pairs.append({
                                "verb": lemma,
                                "dobj": child.lemma_.lower(),
                                "sentence": token.sent.text.strip(),
                            })
        else:
            # Heuristic verb-object approximation for offline test suite
            words = text.lower().split()
            simple_verbs = target_lemmas or {"eat", "fix", "repair", "pivot", "assemble", "buy", "watch", "break"}
            for i, w in enumerate(words):
                clean_w = re.sub(r"[^\w]", "", w)
                matched_verb = None
                if clean_w in simple_verbs:
                    matched_verb = clean_w
                elif clean_w.endswith("s") and clean_w[:-1] in simple_verbs:
                    matched_verb = clean_w[:-1]
                elif clean_w.endswith("es") and clean_w[:-2] in simple_verbs:
                    matched_verb = clean_w[:-2]

                if matched_verb and i + 1 < len(words):
                    next_word = re.sub(r"[^\w]", "", words[i + 1])
                    if next_word not in {"the", "a", "an", "this", "my", "your"}:
                        pairs.append({"verb": matched_verb, "dobj": next_word, "sentence": text})
                    elif i + 2 < len(words):
                        dobj_word = re.sub(r"[^\w]", "", words[i + 2])
                        pairs.append({"verb": matched_verb, "dobj": dobj_word, "sentence": text})

        return pairs
