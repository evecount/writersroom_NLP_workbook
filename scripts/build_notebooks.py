"""
Notebook Generator for Project Writers' Room
============================================
Generates all 7 pedagogically sequenced Jupyter Notebooks with full
code cells, markdown theory, LaTeX equations, interactive exercises,
and autograde checks.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_cell(cell_type: str, source: list[str]) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source],
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook(cells: list[dict], filename: str):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {path}")


# ==========================================
# 01_script_cleaning_dry.ipynb
# ==========================================
def build_nb_01():
    cells = [
        create_cell("markdown", [
            "# Project Writers' Room: Module 1",
            "## Ingestion, Regex & Text Processing Hygiene",
            "",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/writersroom_NLP_workbook/blob/main/notebooks/01_script_cleaning_dry.ipynb)",
            "",
            "### Learning Objectives",
            "1. Deconstruct raw, messy screenplay transcripts into clean dialogue records.",
            "2. Apply Regular Expressions (`re`) to extract scenes, isolate speaker labels, and separate parentheticals.",
            "3. Enforce the **DRY (Don't Repeat Yourself)** software engineering principle by refactoring parsing routines into a reusable `ScriptNormalizer` class."
        ]),
        create_cell("code", [
            "import re",
            "import json",
            "from pathlib import Path",
            "import pandas as pd",
            "import sys",
            "sys.path.insert(0, '..')",
            "",
            "from writersroom import ScriptNormalizer",
            "print('Environment initialized successfully!')"
        ]),
        create_cell("markdown", [
            "### 1. The Screenplay Parsing Problem",
            "Television screenplays do not conform to standardized data schemas. They interleave:",
            "- Scene headings (e.g., `SCENE 1: CENTRAL PERK`)",
            "- Bracketed stage directions (e.g., `[Scene: Ross looks miserable.]`)",
            "- Parenthetical emotional delivery tags (e.g., `(sarcastically)`)",
            "- Uppercase character tags (e.g., `CHANDLER: ...`)"
        ]),
        create_cell("code", [
            "# Inspect a snippet of raw Friends dialogue",
            "raw_sample = Path('../data/raw/friends_sample_raw.txt').read_text(encoding='utf-8')[:600]",
            "print(raw_sample)"
        ]),
        create_cell("markdown", [
            "### 2. Formulating Regex Extraction Rules",
            "We construct regex patterns:",
            "1. `SCENE_HEADER`: `^(?:SCENE\\s+(\\d+)[\\:\\s\\-]*)(.*)$`",
            "2. `STAGE_DIRECTIONS`: `\\[.*?\\]`",
            "3. `PARENTHETICALS`: `\\((.*?)\\)`",
            "4. `SPEAKER_TAG`: `^([A-Z0-9\\s\\.\\'\\-]+)\\:\\s*(.*)$`"
        ]),
        create_cell("code", [
            "normalizer = ScriptNormalizer(default_show='Friends')",
            "records = normalizer.parse_file('../data/raw/friends_sample_raw.txt', show_name='Friends')",
            "df = normalizer.to_dataframe(records)",
            "print(f'Parsed {len(df)} dialogue lines.')",
            "df.head()"
        ]),
        create_cell("markdown", [
            "### 3. Student Hands-on Exercise: Parenthetical Emotion Mining",
            "**Task:** Write a function `analyze_character_parentheticals(df)` that aggregates which characters are given the most explicit stage direction instructions (e.g., 'sarcastically', 'explaining', 'smiling')."
        ]),
        create_cell("code", [
            "# TODO: Student Exercise",
            "def analyze_character_parentheticals(dialogue_df):",
            "    # Filter rows with non-empty parentheticals",
            "    has_paren = dialogue_df[dialogue_df['parenthetical'].str.len() > 0]",
            "    return has_paren.groupby('speaker')['parenthetical'].value_counts()",
            "",
            "emotion_summary = analyze_character_parentheticals(df)",
            "print(emotion_summary)"
        ]),
        create_cell("markdown", [
            "### 4. Self-Verification & Autograding",
            "Run the local autograder to verify your Module 1 implementation."
        ]),
        create_cell("code", [
            "from tests.autograde_checks import check_module_1",
            "check_module_1(normalizer)"
        ])
    ]
    build_notebook(cells, "01_script_cleaning_dry.ipynb")


# ==========================================
# 02_colloquial_tokenization.ipynb
# ==========================================
def build_nb_02():
    cells = [
        create_cell("markdown", [
            "# Project Writers' Room: Module 2",
            "## Tokenization & The Colloquial Speech Problem",
            "",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/writersroom_NLP_workbook/blob/main/notebooks/02_colloquial_tokenization.ipynb)",
            "",
            "### Learning Objectives",
            "1. Understand why standard whitespace tokenizers fail on conversational colloquialisms and contractions.",
            "2. Implement rule-based contraction expansion for sitcom speech ('gonna', 'dunno', 'howdy').",
            "3. Simulate Byte-Pair Encoding (BPE) subword segmentation to prevent iconic catchphrases ('Bazinga', 'Pivot') from becoming `<UNK>` tokens.",
            "4. Measure Out-Of-Vocabulary (OOV) rates across varying vocabulary sizes."
        ]),
        create_cell("code", [
            "import sys",
            "sys.path.insert(0, '..')",
            "from writersroom import ColloquialTokenizer",
            "import pandas as pd",
            "",
            "tok = ColloquialTokenizer()",
            "print('Tokenizer module loaded.')"
        ]),
        create_cell("markdown", [
            "### 1. Standard Tokenization vs. Conversational Speech",
            "Notice how standard whitespace splitting preserves contractions or fragments them awkwardly."
        ]),
        create_cell("code", [
            "sentence = \"Joey: C'mon, you're gonna eat pizza and we ain't gonna stop!\"",
            "print('Basic Tokenizer:', tok.tokenize_basic(sentence))",
            "print('Colloquial Tokenizer (Expanded):', tok.tokenize_colloquial(sentence))"
        ]),
        create_cell("markdown", [
            "### 2. Subwords and Catchphrases: Why BPE Matters",
            "Iconic catchphrases like *Bazinga* or *Pivot* are rarely seen in general English corpora. In standard word-level models, they become `<UNK>`. Subword segmentation preserves morphemes."
        ]),
        create_cell("code", [
            "catchphrases = ['Bazinga', 'Paleontology', 'Multiverse', 'Unpredictable']",
            "for cp in catchphrases:",
            "    print(f'{cp:15} -> {tok.simulate_bpe_subwords(cp)}')"
        ]),
        create_cell("markdown", [
            "### 3. Out-Of-Vocabulary (OOV) Rate Analysis",
            "We train a vocabulary on Season 1 dialogue and evaluate OOV tokens on unseen scenes."
        ]),
        create_cell("code", [
            "train_tokens = tok.tokenize_colloquial('Chandler drinks coffee in central perk with Monica and Joey.')",
            "test_tokens = tok.tokenize_colloquial('Joey says bazinga while eating Swedish flat-pack furniture.')",
            "oov_result = tok.analyze_oov(train_tokens, test_tokens)",
            "print('OOV Analysis Result:', oov_result)"
        ]),
        create_cell("markdown", [
            "### 4. Student Exercise & Autograding",
            "Verify your tokenization logic with the Module 2 autograder test."
        ]),
        create_cell("code", [
            "from tests.autograde_checks import check_module_2",
            "check_module_2(tok)"
        ])
    ]
    build_notebook(cells, "02_colloquial_tokenization.ipynb")


# ==========================================
# 03_lexical_richness_zipf.ipynb
# ==========================================
def build_nb_03():
    cells = [
        create_cell("markdown", [
            "# Project Writers' Room: Module 3",
            "## Lexical Diversity & Vocabulary Saturation",
            "",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/writersroom_NLP_workbook/blob/main/notebooks/03_lexical_richness_zipf.ipynb)",
            "",
            "### Learning Objectives",
            "1. Differentiate between Types ($V$) and Tokens ($N$).",
            "2. Compute standard Type-Token Ratio (TTR) and sample-size invariant Root TTR ($V / \\sqrt{N}$).",
            "3. Mine Hapax Legomena (terms occurring exactly once) to isolate signature character jargon.",
            "4. Validate Zipf's Law on sitcom dialogue through log-log linear regression."
        ]),
        create_cell("code", [
            "import sys",
            "sys.path.insert(0, '..')",
            "from writersroom import LexicalAnalyzer, ColloquialTokenizer",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "df = pd.read_json('../data/processed/dialogue_master.json')",
            "tok = ColloquialTokenizer()",
            "print(f'Loaded {len(df)} total dialogue lines.')"
        ]),
        create_cell("markdown", [
            "### 1. Character Vocabulary Breadth Comparison",
            "We isolate tokens spoken by **Sheldon Cooper** vs. **Joey Tribbiani** vs. **Chandler Bing**."
        ]),
        create_cell("code", [
            "def get_character_tokens(speaker_name):",
            "    lines = df[df['speaker'] == speaker_name]['line']",
            "    text = ' '.join(lines)",
            "    return tok.tokenize_colloquial(text)",
            "",
            "sheldon_tokens = get_character_tokens('SHELDON')",
            "joey_tokens = get_character_tokens('JOEY')",
            "chandler_tokens = get_character_tokens('CHANDLER')",
            "",
            "print(f'Sheldon  Tokens: {len(sheldon_tokens)}, Types: {len(set(sheldon_tokens))}, TTR: {LexicalAnalyzer.compute_ttr(sheldon_tokens)}')",
            "print(f'Joey     Tokens: {len(joey_tokens)}, Types: {len(set(joey_tokens))}, TTR: {LexicalAnalyzer.compute_ttr(joey_tokens)}')",
            "print(f'Chandler Tokens: {len(chandler_tokens)}, Types: {len(set(chandler_tokens))}, TTR: {LexicalAnalyzer.compute_ttr(chandler_tokens)}')"
        ]),
        create_cell("markdown", [
            "### 2. Hapax Legomena: Academic Jargon vs. Everyday Speech",
            "Hapaxes ($f=1$) reveal idiosyncratic character lexicon."
        ]),
        create_cell("code", [
            "print('Sheldon Hapaxes (sample):', LexicalAnalyzer.find_hapaxes(sheldon_tokens)[:10])",
            "print('Joey Hapaxes (sample):', LexicalAnalyzer.find_hapaxes(joey_tokens)[:10])"
        ]),
        create_cell("markdown", [
            "### 3. Zipf's Law: Empirical Log-Log Fit",
            "Zipf's law states that word frequency is inversely proportional to rank:",
            "$$f(r) \\propto \\frac{1}{r^s} \\iff \\log f = \\log C - s \\log r$$"
        ]),
        create_cell("code", [
            "all_tokens = tok.tokenize_colloquial(' '.join(df['line']))",
            "zipf_data = LexicalAnalyzer.fit_zipf_law(all_tokens)",
            "print(f'Fitted Slope: {zipf_data[\"slope\"]}, Intercept: {zipf_data[\"intercept\"]}, R^2: {zipf_data[\"r_squared\"]}')",
            "",
            "# Log-Log Plot",
            "import numpy as np",
            "plt.figure(figsize=(8, 5))",
            "plt.scatter(np.log(zipf_data['ranks'][:50]), np.log(zipf_data['freqs'][:50]), color='crimson', label='Empirical Tokens')",
            "plt.title(\"Zipf's Law in Sitcom Corpus (Log-Log Scale)\")",
            "plt.xlabel('Log(Rank)')",
            "plt.ylabel('Log(Frequency)')",
            "plt.grid(True, alpha=0.3)",
            "plt.legend()",
            "plt.show()"
        ]),
        create_cell("markdown", [
            "### 4. Self-Verification & Autograding"
        ]),
        create_cell("code", [
            "from tests.autograde_checks import check_module_3",
            "check_module_3()"
        ])
    ]
    build_notebook(cells, "03_lexical_richness_zipf.ipynb")


# ==========================================
# 04_stylometrics_cfd.ipynb
# ==========================================
def build_nb_04():
    cells = [
        create_cell("markdown", [
            "# Project Writers' Room: Module 4",
            "## Stylometric Fingerprinting via Conditional Frequencies",
            "",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/writersroom_NLP_workbook/blob/main/notebooks/04_stylometrics_cfd.ipynb)",
            "",
            "### Learning Objectives",
            "1. Construct Conditional Frequency Distributions (CFD) where Condition = Character and Events = Discourse Markers.",
            "2. Normalize frequency rates per 1,000 words for unbiased character comparison.",
            "3. Discover multi-word catchphrases using Pointwise Mutual Information (PMI) bigram collocations without hardcoded keywords.",
            "4. Trace narrative lexical dispersion across script timelines."
        ]),
        create_cell("code", [
            "import sys",
            "sys.path.insert(0, '..')",
            "from writersroom import StylometricFingerprinter, ColloquialTokenizer",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "df = pd.read_json('../data/processed/dialogue_master.json')",
            "tok = ColloquialTokenizer()",
            "print('Loaded processed dialogue.')"
        ]),
        create_cell("markdown", [
            "### 1. Constructing the Character CFD Matrix",
            "We inspect modal verbs and characteristic speech markers ('technically', 'literally', 'honestly', 'bazinga', 'pivot', 'break')."
        ]),
        create_cell("code", [
            "markers = ['technically', 'literally', 'honestly', 'bazinga', 'pivot', 'break', 'could', 'coffee']",
            "cfd_raw = StylometricFingerprinter.build_character_cfd(df, markers)",
            "cfd_norm = StylometricFingerprinter.compute_normalized_marker_rates(cfd_raw)",
            "print('Normalized Marker Rates (per 1,000 tokens):')",
            "cfd_norm"
        ]),
        create_cell("markdown", [
            "### 2. Unsupervised Catchphrase Mining with PMI Collocations",
            "Pointwise Mutual Information measures how much more likely words co-occur than by chance:",
            "$$\\text{PMI}(w_1, w_2) = \\log_2 \\frac{P(w_1, w_2)}{P(w_1) P(w_2)}$$"
        ]),
        create_cell("code", [
            "all_tokens = tok.tokenize_colloquial(' '.join(df['line']))",
            "collocations = StylometricFingerprinter.extract_bigram_collocations_pmi(all_tokens, min_freq=2, top_n=10)",
            "print('Discovered Top Bigram Collocations:')",
            "pd.DataFrame(collocations)"
        ]),
        create_cell("markdown", [
            "### 3. Student Exercise & Autograding",
            "Run check_module_4 to verify your CFD matrix and collocation calculations."
        ]),
        create_cell("code", [
            "from tests.autograde_checks import check_module_4",
            "check_module_4()"
        ])
    ]
    build_notebook(cells, "04_stylometrics_cfd.ipynb")


# ==========================================
# 05_syntax_spacy_matcher.ipynb
# ==========================================
def build_nb_05():
    cells = [
        create_cell("markdown", [
            "# Project Writers' Room: Module 5",
            "## Syntactic Structure & Rule-Based Matchers (spaCy)",
            "",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/writersroom_NLP_workbook/blob/main/notebooks/05_syntax_spacy_matcher.ipynb)",
            "",
            "### Learning Objectives",
            "1. Leverage Part-of-Speech (POS) tagging and dependency trees for comedic dialogue analysis.",
            "2. Formalize comedic rhetorical patterns using rule-based matchers (Chandler's cadence: `Could + PRON/NOUN + BE + ADV* + ADJ?`).",
            "3. Traverse dependency trees to extract Verb $\\rightarrow$ Direct Object (`dobj`) pairs to map character behavioral habits."
        ]),
        create_cell("code", [
            "import sys",
            "sys.path.insert(0, '..')",
            "from writersroom import SitcomSyntaxParser",
            "import pandas as pd",
            "",
            "parser = SitcomSyntaxParser()",
            "print('Syntax parser initialized.')"
        ]),
        create_cell("markdown", [
            "### 1. Chandler's Rhetorical Cadence Detector",
            "Chandler Bing is known for his signature delivery: *\"Could this day BE any more depressing?\"*"
        ]),
        create_cell("code", [
            "lines = [",
            "    'Could this coffee BE any hotter?',",
            "    'Could that Allen wrench BE any more useless?',",
            "    'Could your defense mechanism BE any more predictable?',",
            "    'Joey, please do not eat the sandwich.'",
            "]",
            "for line in lines:",
            "    cadences = parser.find_rhetorical_cadences(line)",
            "    if cadences:",
            "        print(f'MATCH: \"{line}\" -> {cadences[0][\"matched_text\"]}')"
        ]),
        create_cell("markdown", [
            "### 2. Dependency Tree Traversal: Direct Object (`dobj`) Extraction",
            "We extract what characters act upon: `Verb -> dobj`."
        ]),
        create_cell("code", [
            "sample_narrative = 'Joey eats pizza and Ross assembles Scandinavian furniture while Chandler drops the sofa.'",
            "dobj_results = parser.extract_verb_direct_objects(sample_narrative)",
            "pd.DataFrame(dobj_results)"
        ]),
        create_cell("markdown", [
            "### 3. Self-Verification & Autograding"
        ]),
        create_cell("code", [
            "from tests.autograde_checks import check_module_5",
            "check_module_5()"
        ])
    ]
    build_notebook(cells, "05_syntax_spacy_matcher.ipynb")


# ==========================================
# 06_sitcom_embeddings.ipynb
# ==========================================
def build_nb_06():
    cells = [
        create_cell("markdown", [
            "# Project Writers' Room: Module 6",
            "## Vector Semantics & Character Tropes (Embeddings)",
            "",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/writersroom_NLP_workbook/blob/main/notebooks/06_sitcom_embeddings.ipynb)",
            "",
            "### Learning Objectives",
            "1. Understand the Distributional Hypothesis: *\"You shall know a word by the company it keeps.\"*",
            "2. Train Word2Vec dense embeddings on tokenized sitcom dialogue.",
            "3. Conduct sitcom vector arithmetic analogies:",
            "   $$\\vec{v}_{\\text{Ross}} - \\vec{v}_{\\text{Paleontology}} + \\vec{v}_{\\text{Physics}} \\approx \\vec{v}_{\\text{Sheldon}}$$",
            "4. Compute personality radar profiles against emotional anchor trait lexicons."
        ]),
        create_cell("code", [
            "import sys",
            "sys.path.insert(0, '..')",
            "from writersroom import SitcomEmbeddingEngine, ColloquialTokenizer",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "df = pd.read_json('../data/processed/dialogue_master.json')",
            "tok = ColloquialTokenizer()",
            "sentences = [tok.tokenize_colloquial(l) for l in df['line'] if len(l) > 3]",
            "",
            "engine = SitcomEmbeddingEngine(vector_size=30, window=4, min_count=1)",
            "engine.train(sentences * 5)",
            "print(f'Trained vector space with {len(engine.vocab)} vocabulary items.')"
        ]),
        create_cell("markdown", [
            "### 1. Vector Arithmetic Analogy",
            "We test vector relationships in sitcom comedy archetypes."
        ]),
        create_cell("code", [
            "sim_words = engine.vector_analogy(positive=['physics', 'sheldon'], negative=['coffee'], top_k=5)",
            "print('Analogous nearest neighbors:', sim_words)"
        ]),
        create_cell("markdown", [
            "### 2. Personality Radar Projections",
            "We project characters onto 5 personality axes: `[neurotic, romantic, cynical, academic, naive]`."
        ]),
        create_cell("code", [
            "characters = ['SHELDON', 'CHANDLER', 'JOEY', 'GEORGE']",
            "radar_profiles = {}",
            "for char in characters:",
            "    char_lines = df[df['speaker'] == char]['line']",
            "    tokens = tok.tokenize_colloquial(' '.join(char_lines))",
            "    char_vec = engine.build_character_vector(tokens)",
            "    radar_profiles[char] = engine.compute_personality_radar(char_vec)",
            "",
            "radar_df = pd.DataFrame(radar_profiles).T",
            "print('Character Personality Radar Scores:')",
            "radar_df"
        ]),
        create_cell("markdown", [
            "### 3. Self-Verification & Autograding"
        ]),
        create_cell("code", [
            "from tests.autograde_checks import check_module_6",
            "check_module_6()"
        ])
    ]
    build_notebook(cells, "06_sitcom_embeddings.ipynb")


# ==========================================
# 07_script_generator.ipynb
# ==========================================
def build_nb_07():
    cells = [
        create_cell("markdown", [
            "# Project Writers' Room: Capstone Module 7",
            "## The Script Generator Engine (Synthesis & Evaluation)",
            "",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evecount/writersroom_NLP_workbook/blob/main/notebooks/07_script_generator.ipynb)",
            "",
            "### Learning Objectives",
            "1. Build Character-Conditioned N-gram Markov models ($P(w_t \\mid w_{t-1}, \\text{Character})$).",
            "2. Apply Laplace additive smoothing and temperature-scaled probabilistic sampling.",
            "3. Generate multi-turn crossover dialogue scenes between characters from different sitcoms.",
            "4. Execute an evaluation harness comparing classical Markov outputs against neural LLM generation across TTR, vocabulary overlap, and mean line cadence."
        ]),
        create_cell("code", [
            "import sys",
            "sys.path.insert(0, '..')",
            "from writersroom import CharacterMarkovEngine",
            "import pandas as pd",
            "",
            "df = pd.read_json('../data/processed/dialogue_master.json')",
            "generator = CharacterMarkovEngine(n=2, laplace_alpha=0.1)",
            "generator.train_on_dataframe(df)",
            "print('Character Markov Engine trained across all sitcom personas!')"
        ]),
        create_cell("markdown", [
            "### 1. Generating Character-Conditioned Lines",
            "We generate lines conditioned on individual characters and starter prompts."
        ]),
        create_cell("code", [
            "print('CHANDLER:', generator.generate_line('CHANDLER', prompt='could this', temperature=0.7, seed=42))",
            "print('SHELDON: ', generator.generate_line('SHELDON', prompt='technically', temperature=0.7, seed=42))",
            "print('JOEY:    ', generator.generate_line('JOEY', prompt='how you', temperature=0.7, seed=42))",
            "print('KRAMER:  ', generator.generate_line('KRAMER', temperature=0.7, seed=42))"
        ]),
        create_cell("markdown", [
            "### 2. Generating a Multiverse Crossover Scene",
            "We stage an automated scene between Sheldon Cooper and Chandler Bing."
        ]),
        create_cell("code", [
            "scene = generator.generate_scene(['SHELDON', 'CHANDLER', 'JOEY'], turns=6, temperature=0.8)",
            "print('--- AUTOMATED CROSSOVER SCENE ---')",
            "for turn in scene:",
            "    print(f\"{turn['speaker']}: {turn['line']}\")"
        ]),
        create_cell("markdown", [
            "### 3. Evaluation Benchmark Harness",
            "Compare lexical richness and character consistency of generated outputs."
        ]),
        create_cell("code", [
            "gen_lines = [t['line'] for t in scene]",
            "ref_tokens = ' '.join(df['line']).lower().split()",
            "eval_report = generator.evaluate_generation(gen_lines, ref_tokens)",
            "print('Evaluation Metrics Report:', eval_report)"
        ]),
        create_cell("markdown", [
            "### 4. Self-Verification & Autograding"
        ]),
        create_cell("code", [
            "from tests.autograde_checks import check_module_7",
            "check_module_7()"
        ])
    ]
    build_notebook(cells, "07_script_generator.ipynb")


if __name__ == "__main__":
    build_nb_01()
    build_nb_02()
    build_nb_03()
    build_nb_04()
    build_nb_05()
    build_nb_06()
    build_nb_07()
    print("All 7 notebooks successfully generated!")
