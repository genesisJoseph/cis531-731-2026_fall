"""MP7 prompt template: label space, verbalizers, system instruction, demonstration format.

The LLM may only answer with one of the letter verbalizers below; classifiers.py reads
only those letters' scores (constrained decoding), so no other label can be produced.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = json.loads((ROOT / "data" / "labels.json").read_text())   # id -> description
LABEL_IDS = list(LABELS)
LETTERS = [chr(ord("A") + i) for i in range(len(LABEL_IDS))]        # verbalizers: A..H
N_LABELS = len(LABEL_IDS)
assert 8 <= N_LABELS <= 12


SYSTEM = ("You classify short descriptions of songs by their main theme. "
          "Answer with exactly one capital letter from the list of themes. Do not explain.")


def format_options():
    return "\n".join(f"{L}) {desc}" for L, desc in zip(LETTERS, LABELS.values()))


def format_demo(text, label):
    return f"Description: {text}\nTheme: {LETTERS[LABEL_IDS.index(label)]}\n\n"


