import csv
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "bts_gold_reference.csv"
OUTPUT = ROOT / "gold_text.csv"

session = requests.Session()
session.headers.update({
    "User-Agent": "CIS731StudentResearch/1.0 (educational project)"
})

def clean(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def wikipedia(title):
    response = session.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": f'intitle:"{title}" BTS song',
            "gsrlimit": 5,
            "prop": "extracts|info",
            "explaintext": 1,
            "inprop": "url",
            "format": "json",
        },
        timeout=25,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    candidates = list(pages.values())

    # Never silently use an uncertain search result.
    for page in candidates:
        page_title = page.get("title", "")
        extract = clean(page.get("extract", ""))
        if (
            title.casefold() in page_title.casefold()
            and "BTS" in extract
            and len(extract) >= 150
        ):
            return extract[:4000], page.get("fullurl", ""), "review"

    return "", "", "missing"

with SOURCE.open(newline="", encoding="utf-8-sig") as f:
    songs = list(csv.DictReader(f))

print("Songs to process:", len(songs))
results = []

for i, song in enumerate(songs, 1):
    title = song["song_title"].strip()
    try:
        description, url, status = wikipedia(title)
    except Exception as e:
        description, url, status = "", "", "error"
        print("  Error:", str(e)[:100])

    results.append({
        "item_id": song["item_id"],
        "song_title": title,
        "text": description,
        "cohort": "",
        "source_url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
    })

    print(f"{i:02d}/{len(songs)} {title}: {status} ({len(description)} chars)")
    time.sleep(0.5)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", OUTPUT)
print("Candidates for review:", sum(r["status"] == "review" for r in results))
print("Missing/errors:", sum(r["status"] != "review" for r in results))
