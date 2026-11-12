"""Stage 2, piece 1: pull a handful of articles out of the Wikivoyage dump.

What this script does:
  1. Opens the compressed dump (.xml.bz2) and reads it page by page ("streaming"),
     so the whole file never has to fit in memory.
  2. Keeps only the articles we want (Dhaka, Bangkok, Phuket, ...).
  3. Skips "redirect" pages, which are empty pointers to other pages.
  4. Saves each article's raw wikitext to stage2/data/raw/<title>.wiki so you can read it,
     plus stage2/data/raw/_index.json listing everything that was saved.

Run (from the project root):  python -m stage2.s1_extract_articles
"""

import bz2                              # reads .bz2 compressed files
import json                             # writes the _index.json file
import xml.etree.ElementTree as ET      # Python's built-in XML parser
from pathlib import Path                # nicer way to work with file paths

from common.paths import safe_filename
from stage2.config import DUMP_FILE, RAW_DIR

# Where the dump is (shared by all stages), and where to put the extracted articles.
DUMP = DUMP_FILE
OUT_DIR = RAW_DIR

# Exact article titles to keep.
WANTED_TITLES = {
    "Thailand",      # country article: visas, money, safety, etiquette
    "Dhaka",
    "Bangkok",
    "Chiang Mai",
    "Phuket",        # the island overview
    "Phuket Town",
    "Patong",
    "Krabi Town",    # "Krabi" itself is only a disambiguation page
    "Ao Nang",
    "Ko Phi Phi",
}

# Big cities are split into district articles named "City/District",
# e.g. "Bangkok/Sukhumvit". Keep every district of these cities.
WANTED_DISTRICT_PREFIXES = ("Dhaka/", "Bangkok/")


def is_wanted(title: str) -> bool:
    """Should we keep the article with this title?"""
    if title in WANTED_TITLES:
        return True

    # str.startswith accepts a tuple and returns True if ANY prefix matches.
    if title.startswith(WANTED_DISTRICT_PREFIXES):
        return True

    return False


def is_redirect(wikitext: str) -> bool:
    """Redirect pages contain nothing but something like "#REDIRECT [[Bangkok/Silom]]"."""
    without_leading_spaces = wikitext.lstrip()
    upper_case = without_leading_spaces.upper()     # catch "#redirect" and "#Redirect" too
    return upper_case.startswith("#REDIRECT")


def iter_pages(dump_path: Path):
    """Yield (title, wikitext) for every real article in the dump, one at a time.

    This is a *generator* (it uses `yield` instead of `return`): the caller gets
    one page, handles it, then asks for the next. That's what keeps memory low.

    Each page in the XML looks roughly like:
        <page>
          <title>Dhaka</title>
          <ns>0</ns>
          <revision><text>...the wikitext...</text></revision>
        </page>
    """
    with bz2.open(dump_path, "rb") as compressed_file:

        # iterparse reads the XML piece by piece and tells us each time an
        # element has been fully read ("end" event).
        for _event, element in ET.iterparse(compressed_file, events=("end",)):

            # Tag names include a namespace, e.g.
            # "{http://www.mediawiki.org/xml/export-0.11/}page", so we check the ending.
            if not element.tag.endswith("}page"):
                continue

            # "{*}" means "any namespace", so we don't have to type the long URL.
            namespace = element.findtext("{*}ns")
            title = element.findtext("{*}title")
            text = element.findtext("{*}revision/{*}text")
            if text is None:
                text = ""

            # Namespace "0" = real articles. Others are talk pages, user pages, templates...
            if namespace == "0":
                yield title, text

            # We're done with this page: free its memory before reading the next one.
            element.clear()


def main():
    # Create stage2/data/raw (and stage2/data/ if needed). exist_ok=True: no error if it already exists.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = []              # info about each article we keep
    redirects_skipped = 0
    total = 0

    for title, text in iter_pages(DUMP):
        total += 1

        if not is_wanted(title):
            continue

        if is_redirect(text):
            redirects_skipped += 1
            continue

        # Save the raw wikitext. encoding="utf-8" matters on Windows: the
        # articles contain Thai and Bengali script.
        out_path = OUT_DIR / (safe_filename(title) + ".wiki")
        out_path.write_text(text, encoding="utf-8")

        saved.append({"title": title, "chars": len(text)})

        # :<35 pads the title to 35 characters, :>8, right-aligns the number
        # in 8 characters with thousands separators. Makes a neat table.
        print(f"saved  {title:<35} {len(text):>8,} chars")

    # Write a small index of what we extracted; the next step reads it.
    index_path = OUT_DIR / "_index.json"
    index_json = json.dumps(saved, indent=2)
    index_path.write_text(index_json, encoding="utf-8")

    print(f"\nScanned {total:,} articles in the dump.")
    print(f"Saved {len(saved)} articles, skipped {redirects_skipped} redirects.")

    # Warn about any wanted title that wasn't found (e.g. a spelling mistake).
    saved_titles = set()
    for article in saved:
        saved_titles.add(article["title"])

    missing = WANTED_TITLES - saved_titles      # set difference: wanted but not found
    if missing:
        print("Not found (check spelling on wikivoyage.org):", sorted(missing))


# This block only runs when you execute the file directly
# (python -m stage2.s1_extract_articles), not when another file imports it.
if __name__ == "__main__":
    main()
