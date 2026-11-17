import json
import re
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "mp9_unlabeled_chunks.jsonl"

# Download real source documents.
dataset = load_dataset(
    "fancyzhx/ag_news",
    split="train",
    streaming=True
)

chunks = []
seen = set()

for row_number, row in enumerate(dataset):
    text = re.sub(r"\s+", " ", row["text"]).strip()

    # Split real source text into passages.
    words = text.split()

    for start in range(0, len(words), 80):
        passage = " ".join(words[start:start + 80])

        if len(passage.split()) < 25:
            continue

        normalized = passage.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        chunks.append({
            "record_id": f"mp9_unlabeled_{len(chunks):04d}",
            "text": passage,
            "source": "ag_news",
            "source_row": row_number,
            "source_split": "train",
            "source_label_used": False
        })

        if len(chunks) == 1000:
            break

    if len(chunks) == 1000:
        break

with open(OUTPUT, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print("Chunks created:", len(chunks))
print("Unique chunks:", len(seen))
print("Saved:", OUTPUT)

assert len(chunks) == 1000, "Fewer than 1000 chunks collected"