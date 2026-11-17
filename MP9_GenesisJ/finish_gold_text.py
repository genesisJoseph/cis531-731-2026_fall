import csv
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote

import requests

FILE = Path("gold_text.csv")

PAGE_NAMES = {
    "Baepsae (Silver Spoon)": "Silver Spoon",
    "RUN": "Run",
    "Serendipity": "Intro: Serendipity",
    "Persona": "Intro: Persona",
    "Shadow": "Interlude: Shadow",
    "Ego": "Outro: Ego",
    "DOPE": "Dope",
    "FIRE": "Fire",
    "Ugh!": "UGH!",
    "We Are Bulletproof: The Eternal": "We are Bulletproof: the Eternal",
    "Yet To Come": "Yet To Come (The Most Beautiful Moment)",
    "Save ME": "Save Me",
    "Mic Drop": "MIC Drop",
}

def clean(text):
    # Keep introductory prose only; exclude lyrics and other sections.
    text = re.split(r"(?m)^==[^=]", text, maxsplit=1)[0]
    text = re.sub(r"\{\{InfoboxSong.*?^\}\}", "", text,
                  flags=re.S | re.M)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.S)
    text = re.sub(r"\[\[([^]|]+\|)?([^]]+)\]\]", r"\2", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"'{2,}", "", text)
    return re.sub(r"\s+", " ", text).strip()

with FILE.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

session = requests.Session()
session.headers["User-Agent"] = "CIS731EducationalResearch/1.0"

for row in rows:
    if row["text"].strip():
        continue

    song = row["song_title"]
    page_name = PAGE_NAMES.get(song)

    if not page_name:
        print("No page mapping:", song)
        continue

    try:
        response = session.get(
            "https://bts.fandom.com/api.php",
            params={
                "action": "parse",
                "page": page_name,
                "prop": "wikitext",
                "format": "json"
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            print(song, "API error:", data["error"])
            continue

        page = data["parse"]
        description = clean(page["wikitext"]["*"])

        if len(description) < 50:
            print(song, "Insufficient description")
            continue

        row["text"] = description[:4000]
        row["source_url"] = (
            "https://bts.fandom.com/wiki/"
            + quote(page["title"].replace(" ", "_"))
        )
        row["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        row["status"] = "review"

        print(song, "OK:", len(description), "characters")

    except Exception as e:
        print(song, "ERROR:", str(e)[:150])

    time.sleep(1)

with FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nRESULTS")
print("Total songs:", len(rows))
print("With text:", sum(bool(r["text"].strip()) for r in rows))
print("Still missing:",
      [r["song_title"] for r in rows if not r["text"].strip()])
