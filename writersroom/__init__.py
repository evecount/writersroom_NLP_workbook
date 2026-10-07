"""
Writers' Room: Deconstructing & Generating Sitcom Dialogue with NLP
===================================================================

A modular pedagogical toolkit for analyzing, fingerprinting, and generating
dialogue from television comedies (Friends, The Big Bang Theory, Seinfeld).
"""

__version__ = "0.1.0"
__author__ = "Gwendalynn Lim"

from writersroom.normalizer import ScriptNormalizer
from writersroom.tokenizer import ColloquialTokenizer
from writersroom.metrics import LexicalAnalyzer
from writersroom.stylometrics import StylometricFingerprinter
from writersroom.syntax import SitcomSyntaxParser
from writersroom.embeddings import SitcomEmbeddingEngine
from writersroom.generator import CharacterMarkovEngine

__all__ = [
    "ScriptNormalizer",
    "ColloquialTokenizer",
    "LexicalAnalyzer",
    "StylometricFingerprinter",
    "SitcomSyntaxParser",
    "SitcomEmbeddingEngine",
    "CharacterMarkovEngine",
]
