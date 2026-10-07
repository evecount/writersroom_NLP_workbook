# Contributing to Project Writers' Room

Thank you for your interest in expanding **Project Writers' Room: Deconstructing & Generating Sitcom Dialogue with NLP**!

We welcome additions of new television sitcom transcripts, NLP analysis modules, and pedagogical improvements.

---

## How to Add New Sitcom Transcripts

1. Place raw, unedited scripts in `data/raw/` with naming format: `<showname>_<season_ep>_raw.txt`.
2. Ensure speaker lines follow standard screenplay tagging:
   ```text
   SCENE 1: THE LIVING ROOM
   [Scene: Jerry and George enter.]
   JERRY: (sarcastically) Have you heard about this?
   ```
3. Run the parser to update processed datasets:
   ```bash
   python -c "
   from writersroom.normalizer import ScriptNormalizer
   norm = ScriptNormalizer()
   recs = norm.parse_file('data/raw/your_show_raw.txt', show_name='YourShow')
   print(f'Parsed {len(recs)} lines')
   "
   ```

---

## Code Quality & Testing

Before submitting a Pull Request, ensure all tests pass:

```bash
# Run unit tests
python -m unittest discover tests

# Run all 7 module autograder verification checks
python tests/autograde_checks.py
```

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
