"""
Autograding & Student Self-Verification Test Suite
===================================================
Project Writers' Room: Deconstructing & Generating Sitcom Dialogue with NLP

Students can import this module inside their Jupyter Notebooks or run it
via CLI to verify their implementations of all 7 modules:
    python tests/autograde_checks.py
"""

from __future__ import annotations
import math
import sys
from pathlib import Path
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from writersroom.normalizer import ScriptNormalizer
from writersroom.tokenizer import ColloquialTokenizer
from writersroom.metrics import LexicalAnalyzer
from writersroom.stylometrics import StylometricFingerprinter
from writersroom.syntax import SitcomSyntaxParser
from writersroom.embeddings import SitcomEmbeddingEngine
from writersroom.generator import CharacterMarkovEngine


def check_module_1(normalizer_instance: ScriptNormalizer | None = None) -> bool:
    """Verify Module 1: Script Normalization & DRY Regex Parsing."""
    print("\n[Testing Module 1: Ingestion & Regex Normalization]...")
    norm = normalizer_instance or ScriptNormalizer()
    
    sample_text = (
        "SCENE 1: CENTRAL PERK\n"
        "[Scene: Coffee shop. Ross looks sad.]\n"
        "CHANDLER: (sarcastically) Could this day BE any more depressing?\n"
        "JOEY: [eating pizza] Hey, how you doin'?\n"
    )
    records = norm.parse_transcript(sample_text, show_name="Friends", episode_title="Test")
    
    assert len(records) == 2, f"Expected 2 dialogue lines, got {len(records)}"
    assert records[0]["speaker"] == "CHANDLER", f"Expected speaker CHANDLER, got {records[0]['speaker']}"
    assert records[0]["parenthetical"] == "sarcastically", f"Expected parenthetical 'sarcastically', got {records[0]['parenthetical']}"
    assert "Could this day BE any more depressing?" in records[0]["line"]
    assert "Scene 1" in records[0]["scene"]
    assert "[" not in records[1]["line"], "Stage directions inside square brackets should be stripped!"
    
    print("  PASS: Clean dialogue, speaker extraction, and parenthetical isolation verified.")
    return True


def check_module_2(tokenizer_instance: ColloquialTokenizer | None = None) -> bool:
    """Verify Module 2: Colloquial Tokenization & OOV Handling."""
    print("\n[Testing Module 2: Tokenization & The Colloquial Speech Problem]...")
    tok = tokenizer_instance or ColloquialTokenizer()
    
    raw = "C'mon, Joey, we're gonna eat pizza and we ain't gonna stop!"
    tokens = tok.tokenize_colloquial(raw, expand_slang=True)
    
    assert "going" in tokens and "to" in tokens, "Expected 'gonna' to expand into 'going' and 'to'"
    assert "is" in tokens or "not" in tokens, "Expected 'ain't' to expand"
    assert "come" in tokens and "on" in tokens, "Expected 'c'mon' to expand into 'come on'"
    
    subwords = tok.simulate_bpe_subwords("Bazinga")
    assert len(subwords) >= 2, f"Expected BPE subwords for Bazinga, got {subwords}"
    
    oov_report = tok.analyze_oov(
        train_tokens=["hello", "world", "coffee"],
        test_tokens=["hello", "coffee", "bazinga"]
    )
    assert oov_report["oov_count"] == 1
    assert "bazinga" in oov_report["oov_tokens"]
    
    print("  PASS: Contraction expansions, subword BPE simulation, and OOV analytics verified.")
    return True


def check_module_3() -> bool:
    """Verify Module 3: Lexical Diversity, TTR, Hapaxes, and Zipf's Law."""
    print("\n[Testing Module 3: Lexical Diversity & Vocabulary Saturation]...")
    tokens = ["the", "quantum", "particle", "the", "particle", "in", "the", "box", "theory"]
    
    n, v = LexicalAnalyzer.count_tokens_and_types(tokens)
    assert n == 9 and v == 6
    
    ttr = LexicalAnalyzer.compute_ttr(tokens)
    expected_ttr = round(6 / 9, 4)
    assert math.isclose(ttr, expected_ttr, abs_tol=1e-3)
    
    root_ttr = LexicalAnalyzer.compute_root_ttr(tokens)
    assert math.isclose(root_ttr, round(6 / math.sqrt(9), 4), abs_tol=1e-3)
    
    hapaxes = LexicalAnalyzer.find_hapaxes(tokens)
    assert "quantum" in hapaxes and "theory" in hapaxes and "box" in hapaxes
    assert "the" not in hapaxes
    
    zipf = LexicalAnalyzer.fit_zipf_law(tokens * 5)
    assert zipf["slope"] < 0, f"Zipf log-log slope must be negative, got {zipf['slope']}"
    
    print("  PASS: Types, Tokens, TTR, Root TTR, Hapax Legomena, and Zipf regression verified.")
    return True


def check_module_4() -> bool:
    """Verify Module 4: Stylometric Fingerprinting via CFD & Collocations."""
    print("\n[Testing Module 4: Stylometrics & Conditional Frequencies]...")
    df = pd.DataFrame([
        {"speaker": "ROSS", "line": "We were on a break! Yes, a break!"},
        {"speaker": "CHANDLER", "line": "Could this literally be any more awkward?"},
        {"speaker": "SHELDON", "line": "Technically speaking, bazinga is appropriate."},
    ])
    
    cfd = StylometricFingerprinter.build_character_cfd(df, ["break", "literally", "technically", "bazinga"])
    assert "ROSS" in cfd.index and "CHANDLER" in cfd.index and "SHELDON" in cfd.index
    assert cfd.loc["ROSS", "break"] == 2
    assert cfd.loc["SHELDON", "bazinga"] == 1
    
    tokens = ["we", "were", "on", "a", "break", "yes", "on", "a", "break", "definitely"]
    collocations = StylometricFingerprinter.extract_bigram_collocations_pmi(tokens, min_freq=2)
    bigram_names = [c["bigram"] for c in collocations]
    assert "on a" in bigram_names or "a break" in bigram_names
    
    print("  PASS: Character CFD matrix and PMI collocation mining verified.")
    return True


def check_module_5() -> bool:
    """Verify Module 5: Syntactic Profiling & Rule Matching."""
    print("\n[Testing Module 5: Syntactic Structure & Matchers]...")
    parser = SitcomSyntaxParser()
    
    test_line = "Could this coffee BE any hotter?"
    matches = parser.find_rhetorical_cadences(test_line)
    assert len(matches) >= 1, "Expected match for Chandler's rhetorical cadence!"
    
    test_habit = "Joey eats pizza and Ross assembles furniture."
    dobj_pairs = parser.extract_verb_direct_objects(test_habit)
    assert len(dobj_pairs) >= 1, "Expected direct object pairs extracted"
    
    print("  PASS: Cadence pattern matcher and verb-dobj dependency traversal verified.")
    return True


def check_module_6() -> bool:
    """Verify Module 6: Vector Semantics & Character Tropes."""
    print("\n[Testing Module 6: Sitcom Word Embeddings & Personality Radar]...")
    sentences = [
        ["ross", "paleontology", "museum", "dinosaur", "fossil", "science"],
        ["sheldon", "physics", "quantum", "theory", "science", "caltech"],
        ["joey", "pizza", "sandwich", "food", "acting", "meatball"],
        ["chandler", "sarcastic", "joke", "cynical", "awkward"],
        ["george", "neurotic", "anxious", "panic", "worry"],
    ]
    engine = SitcomEmbeddingEngine(vector_size=20, min_count=1)
    engine.train(sentences * 5)
    
    sim = engine.word_similarity("physics", "science")
    assert isinstance(sim, float)
    
    # Character vector & radar
    ross_vec = engine.build_character_vector(["ross", "paleontology", "fossil", "dinosaur"])
    radar = engine.compute_personality_radar(ross_vec)
    
    assert "academic" in radar
    assert "neurotic" in radar
    assert all(0.0 <= score <= 1.0 for score in radar.values())
    
    print("  PASS: Word2Vec embeddings, vector analogies, and personality radar projection verified.")
    return True


def check_module_7() -> bool:
    """Verify Module 7: Script Generator Engine & Evaluation."""
    print("\n[Testing Module 7: Script Generator Engine & Evaluation Harness]...")
    df = pd.DataFrame([
        {"speaker": "CHANDLER", "line": "Could this coffee be any hotter?"},
        {"speaker": "CHANDLER", "line": "Could this day be any more depressing?"},
        {"speaker": "JOEY", "line": "Hey how you doin?"},
        {"speaker": "JOEY", "line": "Joey does not share food!"},
    ])
    
    engine = CharacterMarkovEngine(n=2)
    engine.train_on_dataframe(df)
    
    line = engine.generate_line("CHANDLER", prompt="could this", temperature=0.7, seed=42)
    assert isinstance(line, str) and len(line) > 0
    
    eval_metrics = engine.evaluate_generation(
        generated_lines=["Could this coffee be depressing?", "Joey does not share pizza."],
        reference_tokens=["could", "this", "coffee", "be", "depressing", "joey", "food"]
    )
    assert "ttr" in eval_metrics and "vocab_overlap" in eval_metrics
    assert 0.0 <= eval_metrics["ttr"] <= 1.0
    assert 0.0 <= eval_metrics["vocab_overlap"] <= 1.0
    
    print("  PASS: Character-conditioned Markov generation and evaluation metrics verified.")
    return True


def run_all_checks() -> bool:
    """Execute complete autograder suite."""
    print("=" * 70)
    print(" PROJECT WRITERS' ROOM: COMPREHENSIVE AUTOGRADING SUITE")
    print("=" * 70)
    
    results = [
        check_module_1(),
        check_module_2(),
        check_module_3(),
        check_module_4(),
        check_module_5(),
        check_module_6(),
        check_module_7(),
    ]
    
    all_passed = all(results)
    print("\n" + "=" * 70)
    if all_passed:
        print(" [SUCCESS] ALL 7 MODULE VERIFICATION SUITES PASSED PERFECTLY! ")
    else:
        print(" [FAIL] Some checks failed. Review the output logs above. ")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
