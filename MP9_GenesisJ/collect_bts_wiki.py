import csv
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "gold_text.csv"
rows = list(csv.DictReader(FILE.open(encoding="utf-8")))

API = "https://bts.fandom.com/api.php"

def clean_wikitext(text):
    # Only use introductory prose before the first section.
    # Exclude lyrics, infoboxes, media, and templates.
    text = re.split(r"(?m)^==[^=]", text, maxsplit=1)[0]
    text = re.sub(r"\{\{InfoboxSong.*?^\}\}", "", text,
                  flags=re.S | re.M)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.S)
    text = re.sub(r"\[\[([^]|]+\|)?([^]]+)\]\]", r"\2", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

session = requests.Session()
session.headers["User-Agent"] = "CIS731EducationalResearch/1.0"

for i, row in enumerate(rows, 1):
    if row["text"].strip():
        print(f"{i:02d} {row['song_title']}: existing Wikipedia candidate")
        continue

    title = row["song_title"]
    try:
        response = session.get(
            API,
            params={
                "action": "parse",
                "page": title,
                "prop": "wikitext",
                "format": "json"
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            print(f"{i:02d} {title}: page not found")
            continue

        page = data["parse"]
        raw = page.get("wikitext", {}).get("*", "")
        description = clean_wikitext(raw)

        if len(description) < 50:
            print(f"{i:02d} {title}: insufficient description")
            continue

        row["text"] = description[:4000]
        row["source_url"] = (
            "https://bts.fandom.com/wiki/"
            + quote(page["title"].replace(" ", "_"))
        )
        row["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        row["status"] = "review"

        print(f"{i:02d} {title}: retrieved {len(description)} chars")

    except Exception as e:
        print(f"{i:02d} {title}: {type(e).__name__}: {e}")

    # Save after each request so progress isn't lost.
    with FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    time.sleep(1)

print("\nTOTAL WITH TEXT:",
      sum(bool(r["text"].strip()) for r in rows),
      "/ 50")
print("Saved:", FILE)
