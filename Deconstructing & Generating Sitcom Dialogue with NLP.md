Here is a complete, pedagogically sequenced **Workbook Project Curriculum** titled:

# ***Project Writers' Room: Deconstructing & Generating Sitcom Dialogue with NLP***

This curriculum is structured as a modular interactive Python workbook (Jupyter/Colab) where students learn fundamental NLP concepts by reverse-engineering, analyzing, and generating sitcom scripts (*Friends*, *The Big Bang Theory*, *Seinfeld*).

## **Pedagogical Architecture & Module Flow**

Raw Screenplays (.txt / .json)  
   │  
   ▼  
\[Module 1: Ingestion & Regex Parsing\] ──► Dialogue Extraction & Speaker Tagging  
   │  
   ▼  
\[Module 2: Tokenization & Normalization\] ──► Contractions, Slang & Custom Vocabularies  
   │  
   ▼  
\[Module 3: Corpus Statistics & Lexical Diversity\] ──► TTR, Hapax Legomena & Zipf's Law  
   │  
   ▼  
\[Module 4: Stylometrics & Conditional Frequencies\] ──► Character Voice Fingerprinting (CFD)  
   │  
   ▼  
\[Module 5: Syntactic Profiling & spaCy\] ──► POS Tagging, Rule Matching & Dependency Parsing  
   │  
   ▼  
\[Module 6: Semantic Clustering & Embeddings\] ──► Vector Spaces, Word2Vec & Character Tropes  
   │  
   ▼  
\[Module 7: The Script Generator Engine\] ──► Markov Chains ➔ N-gram LM ➔ Neural Fine-Tuning

## **Module Breakdown & Student Deliverables**

### **Module 1: Ingestion, Regex & Text Processing Hygiene**

* **NLP Concepts:** Regular expressions, string encoding, text normalization, the DRY (Don't Repeat Yourself) principle.  
* **The Sitcom Task:**  
  * Raw script files mix stage directions (\[entering Central Perk\]), scene headings (SCENE 1: APARTMENT), and dialogue lines (CHANDLER: Could this be any more...).  
  * Students design robust regex pipelines to split transcripts into a clean tabular structure: \[Episode, Scene, Speaker, Line, Parenthetical\].  
* **Hands-on Student Exercise:** Refactor duplicate cleaning routines across multiple seasons into a single modular ScriptNormalizer class.

### **Module 2: Tokenization & The Colloquial Speech Problem**

* **NLP Concepts:** Word-level vs. Subword-level (BPE) tokenization, sentence splitting, contraction expansion, Out-of-Vocabulary (OOV) handling.  
* **The Sitcom Task:**  
  * Sitcoms are packed with conversational elisions (*"gonna"*, *"dunno"*, *"howdy"*) and idiosyncratic catchphrases (*"Bazinga"*, *"Pivot"*).  
  * Students evaluate standard whitespace tokenizers vs. rule-based tokenizers vs. BPE.  
* **Hands-on Student Exercise:** Inspect tokenization splits for comedic dialogue and determine how subword segmentation prevents catchphrases from becoming unknown (\<UNK\>) tokens.

### **Module 3: Lexical Diversity & Vocabulary Saturation**

* **NLP Concepts:** Types vs. Tokens (\$V\$ vs. \$N\$), Type-Token Ratio (TTR), Root TTR, Hapax Legomena, Cumulative Frequency Distributions, Zipf's Law.  
* **The Sitcom Task:**  
  * Rank characters by vocabulary breadth.  
  * Contrast Sheldon Cooper (high TTR, dense academic hapaxes) against Joey Tribbiani (lower TTR, reliance on a compact set of conversational high-frequency tokens).  
* **Hands-on Student Exercise:**  
  * Plot the **Cumulative Frequency Curves** of different characters on the same axis to visualize who monopolizes rare words vs. who repeats high-frequency functional words.  
  * Compute fdist.hapaxes() to identify each character's signature unique terminology.

### **Module 4: Stylometric Fingerprinting via Conditional Frequencies**

* **NLP Concepts:** FreqDist, ConditionalFreqDist (CFD), Dispersion Plots, Collocation Analysis (Pointwise Mutual Information \- PMI).  
* **The Sitcom Task:**  
  * Build a 2D matrix where **Condition \= Character** and **Events \= Characteristic Modals / Discourse Markers** (*"literally"*, *"honestly"*, *"technically"*, *"suppose"*).  
  * Generate an NLTK **Dispersion Plot** to trace character story presence and phrase spikes across a 24-episode season.  
* **Hands-on Student Exercise:** Use Bigram Collocation extraction to discover multi-word catchphrases automatically (e.g., detecting *“we were on a break”* without hardcoded string matching).

### **Module 5: Syntactic Structure & Rule-Based Matchers (spaCy)**

* **NLP Concepts:** Part-of-Speech (POS) Tagging, Dependency Trees (Head vs. Dependent), Rule-Based Pattern Matching (spacy.matcher.Matcher).  
* **The Sitcom Task:**  
  * Analyze joke cadence through syntactic patterns.  
  * Use spaCy's Matcher to extract rhetorical patterns:  
    * Chandler's Rhetorical Cadence: \[Could\] \+ \[PRON\] \+ \[BE\] \+ \[ADV\] \+ \[ADJ\]?  
    * Sarcastic Retorts: Interrogatives immediately governed by negation verbs.  
* **Hands-on Student Exercise:** Write a dependency parser traversal that extracts all direct objects (dobj) associated with specific action verbs for each character to map their recurring behavioral habits.

### **Module 6: Vector Semantics & Character Tropes (Embeddings)**

* **NLP Concepts:** The Distributional Hypothesis, One-Hot vs. Dense Representations, Word2Vec (CBOW vs. Skip-gram), Cosine Similarity.  
* **The Sitcom Task:**  
  * Train a domain-specific Word2Vec model on the combined transcripts.  
  * Test semantic vector arithmetic on sitcom archetypes:  
    \$\$\\vec{v}\_{\\text{Ross}} \- \\vec{v}\_{\\text{Paleontology}} \+ \\vec{v}\_{\\text{Physics}} \\approx \\vec{v}\_{\\text{Sheldon}}\$\$  
* **Hands-on Student Exercise:** Calculate the cosine similarity of character dialogue vectors against a set of emotional anchor words (*neurotic*, *romantic*, *cynical*, *naive*) to build automatic personality radar charts.

### **Module 7: The Script Generator Engine (Synthesis & Evaluation)**

* **NLP Concepts:** Markov Chains, \$N\$-gram Language Modeling with Laplace / Good-Turing Smoothing, Perplexity, Prompt Conditioning.  
* **The Sitcom Task:**  
  * **Level 1 (Classical Baseline):** Build a Character-Conditioned Bigram/Trigram Markov generator using CFDs:  
    \$\$P(w\_t \\mid w\_{t-1}, \\text{Character})\$\$  
  * **Level 2 (Neural Comparison):** Feed character dialogue prompts into an open-weight small model (e.g., Llama/Mistral via Hugging Face or Ollama) with temperature variations.  
* **Hands-on Student Exercise:** Run an evaluation matrix comparing the classical Markov outputs against neural outputs on metrics of:  
  * Vocabulary diversity (TTR)  
  * Character consistency (cosine similarity against baseline character embeddings)  
  * Syntactic validity (parse error rate)

## **Structure of the Student Workbook Repository**

nlp-writers-room/  
│  
├── README.md                          \# Course overview, prerequisites & setup guide  
├── environment.yml                    \# Conda reproducibility manifest (Python, spaCy, NLTK, etc.)  
│  
├── data/  
│   ├── raw/                           \# Raw script transcripts (.txt)  
│   └── processed/                     \# Parsed parquet/json dialogue datasets  
│  
├── notebooks/  
│   ├── 01\_script\_cleaning\_dry.ipynb   \# Module 1  
│   ├── 02\_colloquial\_tokenization.ipynb\# Module 2  
│   ├── 03\_lexical\_richness\_zipf.ipynb \# Module 3  
│   ├── 04\_stylometrics\_cfd.ipynb      \# Module 4  
│   ├── 05\_syntax\_spacy\_matcher.ipynb  \# Module 5  
│   ├── 06\_sitcom\_embeddings.ipynb     \# Module 6  
│   └── 07\_script\_generator.ipynb      \# Capstone Module 7  
│  
└── tests/  
    └── autograde\_checks.py            \# Unit tests for students to verify assignments locally

This creates an applied, end-to-end framework where every single line of standard syllabus theory—from basic cleaning and token counts to dependency trees and vector math—directly drives a fun, recognizable output.