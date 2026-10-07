"""
Module 1: Ingestion, Regex & Text Processing Hygiene
=====================================================
Modular ScriptNormalizer implementing DRY principles to convert messy,
inconsistent screenplay text files into clean, structured dialogue dataframes.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class ScriptNormalizer:
    """
    Regex-driven screenplay parser and cleaner.
    
    Transforms heterogeneous transcripts containing stage directions,
    scene headings, parentheticals, and uppercase speaker tags into
    a unified tabular schema: [show, episode, scene, speaker, line, parenthetical].
    """

    # Regex definitions
    SCENE_HEADER_PATTERN = re.compile(
        r"^(?:SCENE\s+(\d+|[IVXLCDM]+)[\:\s\-]*)(.*)$",
        re.IGNORECASE | re.MULTILINE
    )
    STAGE_DIRECTION_BRACKETS = re.compile(r"\[.*?\]", re.DOTALL)
    PARENTHETICAL_PATTERN = re.compile(r"\((.*?)\)")
    SPEAKER_LINE_PATTERN = re.compile(
        r"^([A-Z][A-Z0-9\s\.\'\-]{1,30})\:\s*(.*)$",
        re.MULTILINE
    )

    def __init__(self, default_show: str = "Sitcom"):
        self.default_show = default_show

    def clean_stage_directions(self, text: str) -> str:
        """Remove stage directions enclosed in square brackets [ ... ]."""
        return self.STAGE_DIRECTION_BRACKETS.sub("", text)

    def extract_parentheticals(self, text: str) -> tuple[str, list[str]]:
        """
        Extract parenthetical stage directions (e.g. '(sarcastically)')
        from a dialogue line, returning the cleaned text and list of parentheticals.
        """
        parentheticals = self.PARENTHETICAL_PATTERN.findall(text)
        cleaned_text = self.PARENTHETICAL_PATTERN.sub("", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text, [p.strip() for p in parentheticals]

    def parse_transcript(
        self,
        raw_text: str,
        show_name: Optional[str] = None,
        episode_title: str = "Pilot"
    ) -> list[dict[str, Any]]:
        """
        Parse raw transcript text into structured dialogue records.

        Returns a list of dictionaries with keys:
            ['show', 'episode', 'scene', 'speaker', 'line', 'parenthetical']
        """
        show = show_name or self.default_show
        records: list[dict[str, Any]] = []
        
        current_scene = "Scene 1: Opening"
        lines = raw_text.splitlines()

        for line_raw in lines:
            line_str = line_raw.strip()
            if not line_str:
                continue

            # Check if this line is a scene heading
            scene_match = self.SCENE_HEADER_PATTERN.match(line_str)
            if scene_match:
                scene_num = scene_match.group(1)
                scene_desc = scene_match.group(2).strip()
                current_scene = f"Scene {scene_num}: {scene_desc}" if scene_desc else f"Scene {scene_num}"
                continue

            # If the entire line is a bracketed stage direction [Scene: ...], skip
            if line_str.startswith("[") and line_str.endswith("]"):
                continue

            # Strip any bracketed directions inside dialogue line
            line_without_brackets = self.clean_stage_directions(line_str)
            if not line_without_brackets.strip():
                continue

            # Check for speaker tag pattern: "CHANDLER: Could this be..."
            speaker_match = self.SPEAKER_LINE_PATTERN.match(line_without_brackets)
            if speaker_match:
                raw_speaker = speaker_match.group(1).strip().upper()
                raw_dialogue = speaker_match.group(2).strip()

                cleaned_dialogue, parentheticals = self.extract_parentheticals(raw_dialogue)
                parenthetical_str = "; ".join(parentheticals) if parentheticals else ""

                if cleaned_dialogue:
                    records.append({
                        "show": show,
                        "episode": episode_title,
                        "scene": current_scene,
                        "speaker": raw_speaker,
                        "line": cleaned_dialogue,
                        "parenthetical": parenthetical_str,
                    })

        return records

    def parse_file(
        self,
        file_path: str | Path,
        show_name: Optional[str] = None,
        episode_title: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Read and parse a screenplay file from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Script file not found: {path}")

        ep_title = episode_title or path.stem.replace("_", " ").title()
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_transcript(raw_text, show_name=show_name, episode_title=ep_title)

    def to_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert parsed dialogue records into a pandas DataFrame."""
        columns = ["show", "episode", "scene", "speaker", "line", "parenthetical"]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records)[columns]

    def export_processed(
        self,
        records: list[dict[str, Any]],
        output_path: str | Path,
        fmt: str = "json"
    ) -> Path:
        """Export parsed dialogues to json or csv."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt.lower() == "json":
            out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        elif fmt.lower() == "csv":
            df = self.to_dataframe(records)
            df.to_csv(out, index=False, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'json' or 'csv'.")
        return out
