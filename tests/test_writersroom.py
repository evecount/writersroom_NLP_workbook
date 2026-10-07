"""
Test Suite for Writers' Room Library (unittest & pytest compatible)
===================================================================
"""

import math
import unittest
import pandas as pd
from writersroom import (
    ScriptNormalizer,
    ColloquialTokenizer,
    LexicalAnalyzer,
    StylometricFingerprinter,
    SitcomSyntaxParser,
    SitcomEmbeddingEngine,
    CharacterMarkovEngine,
)


class TestWritersRoom(unittest.TestCase):

    def test_normalizer_dry(self):
        norm = ScriptNormalizer()
        raw = (
            "SCENE 1: THE COFFEE SHOP\n"
            "[Scene: Central Perk.]\n"
            "CHANDLER: (sarcastically) Could this day BE any more depressing?\n"
        )
        records = norm.parse_transcript(raw, show_name="Friends")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["speaker"], "CHANDLER")
        self.assertEqual(records[0]["parenthetical"], "sarcastically")
        self.assertIn("Scene 1", records[0]["scene"])

    def test_tokenizer_colloquial_and_oov(self):
        tok = ColloquialTokenizer()
        text = "We gotta go, Joey dunno how to fix it!"
        tokens = tok.tokenize_colloquial(text)
        self.assertIn("got", tokens)
        self.assertIn("to", tokens)
        self.assertIn("know", tokens)

        oov = tok.analyze_oov(["the", "big", "bang"], ["the", "bazinga"])
        self.assertEqual(oov["oov_count"], 1)
        self.assertIn("bazinga", oov["oov_tokens"])

    def test_lexical_analyzer_zipf_and_ttr(self):
        tokens = ["coffee", "coffee", "cup", "central", "perk", "coffee"]
        ttr = LexicalAnalyzer.compute_ttr(tokens)
        self.assertTrue(0.0 < ttr <= 1.0)

        root_ttr = LexicalAnalyzer.compute_root_ttr(tokens)
        self.assertTrue(root_ttr > 0)

        hapaxes = LexicalAnalyzer.find_hapaxes(tokens)
        self.assertIn("cup", hapaxes)
        self.assertNotIn("coffee", hapaxes)

        zipf = LexicalAnalyzer.fit_zipf_law(tokens * 3)
        self.assertLess(zipf["slope"], 0)

    def test_stylometrics_cfd_and_pmi(self):
        df = pd.DataFrame([
            {"speaker": "CHANDLER", "line": "Could this literally be awkward?"},
            {"speaker": "SHELDON", "line": "Technically, bazinga is logical."},
        ])
        cfd = StylometricFingerprinter.build_character_cfd(df, ["literally", "technically"])
        self.assertEqual(cfd.loc["CHANDLER", "literally"], 1)
        self.assertEqual(cfd.loc["SHELDON", "technically"], 1)

        norm_rates = StylometricFingerprinter.compute_normalized_marker_rates(cfd)
        self.assertIn("literally", norm_rates.columns)

    def test_syntax_chandler_cadence(self):
        parser = SitcomSyntaxParser()
        res = parser.find_rhetorical_cadences("Could this day BE any more depressing?")
        self.assertGreaterEqual(len(res), 1)
        self.assertEqual(res[0]["pattern"], "CHANDLER_RHETORICAL")

    def test_embeddings_and_radar(self):
        sentences = [
            ["ross", "paleontology", "museum", "science"],
            ["sheldon", "physics", "quantum", "science"],
            ["george", "neurotic", "anxious", "panic"],
        ]
        engine = SitcomEmbeddingEngine(vector_size=10, min_count=1)
        engine.train(sentences * 3)
        
        vec = engine.build_character_vector(["sheldon", "physics"])
        self.assertGreater(len(vec), 0)
        radar = engine.compute_personality_radar(vec)
        self.assertIn("academic", radar)

    def test_markov_generator(self):
        df = pd.DataFrame([
            {"speaker": "JOEY", "line": "How you doin?"},
            {"speaker": "JOEY", "line": "Joey does not share food!"},
        ])
        engine = CharacterMarkovEngine(n=2)
        engine.train_on_dataframe(df)
        gen = engine.generate_line("JOEY", prompt="how you", temperature=0.7)
        self.assertIsInstance(gen, str)
        self.assertGreater(len(gen), 0)


if __name__ == "__main__":
    unittest.main()
