"""
GLaDOS voice line scraper — Portal Wiki
Downloads all GLaDOS audio files from theportalwiki.com.

Sources:
  - GLaDOS voice lines (Portal)
  - GLaDOS voice lines (Portal 2)
  - GLaDOS voice lines (Cooperative Testing Initiative)

Output: raw_audio/ directory with .ogg files + metadata.json
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WIKI_BASE = "https://theportalwiki.com"
SOURCES = [
    "/wiki/GLaDOS_voice_lines_(Portal)",
    "/wiki/GLaDOS_voice_lines_(Portal_2)",
    "/wiki/GLaDOS_voice_lines_(Cooperative_Testing_Initiative)",
    "/wiki/GLaDOS_voice_lines_(Other)",
]

OUT_DIR = Path("raw_audio")
OUT_DIR.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "X1-GLaDOS-VoiceTrainer/1.0"


def get_audio_links(wiki_path: str) -> list[dict]:
    """Scrape a wiki voice lines page and return list of {filename, transcript, url}."""
    url = WIKI_BASE + wiki_path
    print(f"Fetching {url}")
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen_urls = set()

    # Links are inside <span> tags; each file appears 3x (Download, icon, Play)
    # Get only "Download" links to deduplicate
    for a in soup.find_all("a", href=re.compile(r"\.(ogg|wav)$", re.I)):
        if a.get_text(strip=True) != "Download":
            continue

        audio_href = a["href"]
        if audio_href.startswith("//"):
            audio_href = "https:" + audio_href
        elif audio_href.startswith("/"):
            audio_href = WIKI_BASE + audio_href

        if audio_href in seen_urls:
            continue
        seen_urls.add(audio_href)

        # Transcript lives in the parent <li> element
        li = a.find_parent("li")
        transcript = ""
        if li:
            # Get text, strip the Download/Play button text
            raw = li.get_text(separator=" ", strip=True)
            # Remove trailing "Download Play" noise
            transcript = re.sub(r"\s*(Download|Play)\s*", " ", raw).strip()
            # Strip surrounding asterisks/quotes used by the wiki
            transcript = transcript.strip("*\"' ")

        filename = Path(audio_href).name
        results.append({
            "filename": filename,
            "transcript": transcript,
            "url": audio_href,
            "source": wiki_path,
        })

    return results


def download_file(url: str, dest: Path) -> bool:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        return True
    try:
        resp = SESSION.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


def main():
    all_entries = []

    for source in SOURCES:
        try:
            entries = get_audio_links(source)
            print(f"  Found {len(entries)} audio files")
            all_entries.extend(entries)
        except Exception as e:
            print(f"  ERROR scraping {source}: {e}")
        time.sleep(1)  # be polite to the wiki

    print(f"\nTotal: {len(all_entries)} files. Downloading...")

    downloaded = 0
    for entry in all_entries:
        dest = OUT_DIR / entry["filename"]
        ok = download_file(entry["url"], dest)
        if ok:
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"  {downloaded}/{len(all_entries)}")
        time.sleep(0.1)

    # Save metadata
    meta_path = OUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\nDone. {downloaded}/{len(all_entries)} downloaded → {OUT_DIR}/")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
