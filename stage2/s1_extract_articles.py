"""Stage 2, piece 1: pull every article of the chosen regions out of the Wikivoyage dump.

HOW ARTICLES ARE CHOSEN: BY RULE, NOT BY NAME
  Every article says which bigger place it belongs to, with {{IsPartOf|...}}.
  Following those links upward gives a chain, e.g.
      Ao Nang -> Krabi Province -> Southern Thailand -> Thailand -> Southeast Asia -> Asia
  An article is wanted if its chain reaches one of WANTED_REGIONS
  (and, optionally, if its quality is at least MIN_QUALITY).

WHY TWO PASSES OVER THE DUMP
  The dump is read top to bottom, and a child often comes BEFORE its parent
  (Ao Nang is article #1,091, Krabi Province #15,741). So when we read Ao Nang
  we can't yet tell whether it's in Thailand.
    Pass 1: read everything, but keep only "title -> parent" (+ quality, redirects).
            A small dictionary, no article text. Then work out the wanted titles.
    Pass 2: read the dump again and save the wanted articles.

OUTPUT
  stage2/data/raw/<title>.wiki   the raw wikitext of each article
  stage2/data/raw/_index.json    the list of saved articles (WikivoyageLoader reads it)

Run (from the project root):  python -m stage2.s1_extract_articles
"""

import bz2                              # reads .bz2 compressed files
import json                             # writes the _index.json file
import re                               # finds templates in the wikitext
import time
import xml.etree.ElementTree as ET      # Python's built-in XML parser
from collections import Counter         # counts things, e.g. articles per quality
from pathlib import Path                # nicer way to work with file paths

from common.paths import safe_filename
from common.wikitext import extract_is_part_of
from stage2.config import DUMP_FILE, RAW_DIR

# Where the dump is (shared by all stages), and where to put the extracted articles.
DUMP = DUMP_FILE
OUT_DIR = RAW_DIR

# --- What to extract ------------------------------------------------------
# Every article whose IsPartOf chain reaches one of these (including the
# region's own article, e.g. "Thailand").
WANTED_REGIONS = ["Thailand", "Bangladesh"]

# Lowest quality to keep: None (everything), "outline", "usable", "guide" or "star".
MIN_QUALITY = None

# Wikivoyage's quality ranking, worst to best. "unknown" = no quality template.
QUALITY_LEVELS = ["unknown", "outline", "usable", "guide", "star"]

# Safety limit when walking up a chain (protects against loops like A -> B -> A).
MAX_CHAIN_LENGTH = 20


# --------------------------------------------------------------------------
# Reading the dump
# --------------------------------------------------------------------------

def iter_pages(dump_path: Path):
    """Yield (title, wikitext) for every page in namespace 0 (real articles), one at a time.

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


# --------------------------------------------------------------------------
# Small helpers that look at one article's wikitext
# --------------------------------------------------------------------------

def normalize_title(name: str) -> str:
    """Make a link target look like a real title: "South_Asia" -> "South Asia".

    Links may use underscores and a lowercase first letter; titles never do.
    """
    name = name.replace("_", " ")
    name = name.strip()
    if name == "":
        return name
    first_letter = name[0].upper()
    return first_letter + name[1:]


def is_redirect(wikitext: str) -> bool:
    """Redirect pages contain nothing but something like "#REDIRECT [[Bangkok/Silom]]"."""
    without_leading_spaces = wikitext.lstrip()
    upper_case = without_leading_spaces.upper()     # catch "#redirect" and "#Redirect" too
    return upper_case.startswith("#REDIRECT")


def redirect_target(wikitext: str) -> str | None:
    """"#REDIRECT [[Krabi (province)#Get in]]" -> "Krabi (province)" (None if not found)."""
    match = re.search(r"\[\[([^\]|#]+)", wikitext)
    if match is None:
        return None
    return normalize_title(match.group(1))


def extract_quality(wikitext: str) -> str:
    """Wikivoyage's quality template, e.g. {{usablecity}} -> "usable", {{outlineregion}} -> "outline".

    Returns "unknown" if the article has no quality template.
    """
    match = re.search(
        r"\{\{\s*(outline|usable|guide|star)"
        r"\s*(city|region|country|park|district|airport|topic|itinerary|diveguide|ruralarea)?"
        r"\s*[|}]",
        wikitext,
        re.IGNORECASE,
    )
    if match is None:
        return "unknown"
    return match.group(1).lower()


# --------------------------------------------------------------------------
# Pass 1: the "family tree" of places
# --------------------------------------------------------------------------

def build_place_tree(dump_path: Path):
    """Pass 1: read the whole dump, keeping only small facts about each page.

    Returns three dictionaries:
        parents:   {"Ao Nang": "Krabi Province", ...}       from {{IsPartOf|...}}
        qualities: {"Ao Nang": "usable", ...}
        redirects: {"Krabi": "Krabi Province", ...}        redirect page -> target
    """
    parents = {}
    qualities = {}
    redirects = {}

    for title, text in iter_pages(dump_path):

        if is_redirect(text):
            target = redirect_target(text)
            if target is not None:
                redirects[title] = target
            continue

        qualities[title] = extract_quality(text)

        parent = extract_is_part_of(text)      # same helper the metadata uses
        if parent is not None:
            parents[title] = normalize_title(parent)

    return parents, qualities, redirects


def find_region(title: str, parents: dict, redirects: dict) -> str | None:
    """Walk up the IsPartOf chain; return the wanted region it reaches, or None.

    "Ao Nang" -> "Krabi Province" -> "Southern Thailand" -> "Thailand"  ->  "Thailand"
    """
    current = title
    seen = set()

    for _step in range(MAX_CHAIN_LENGTH):
        if current in WANTED_REGIONS:
            return current

        if current in seen:            # a loop (A -> B -> A): give up
            return None
        seen.add(current)

        parent = parents.get(current)
        if parent is None:             # top of the chain, no wanted region found
            return None

        # The parent link may point at a redirect page: follow it to the real title.
        current = redirects.get(parent, parent)

    return None


def is_good_enough(quality: str) -> bool:
    """True if the quality is at least MIN_QUALITY (always True when MIN_QUALITY is None)."""
    if MIN_QUALITY is None:
        return True
    article_rank = QUALITY_LEVELS.index(quality)      # e.g. "usable" -> 2
    minimum_rank = QUALITY_LEVELS.index(MIN_QUALITY)
    return article_rank >= minimum_rank


def choose_titles(parents: dict, qualities: dict, redirects: dict) -> dict:
    """Apply the rules to every article. Returns {title: region} for the wanted ones."""
    chosen = {}
    too_low_quality = 0

    for title in qualities:                           # every real (non-redirect) article
        region = find_region(title, parents, redirects)
        if region is None:
            continue

        if not is_good_enough(qualities[title]):
            too_low_quality += 1
            continue

        chosen[title] = region

    if too_low_quality > 0:
        print(f"  left out {too_low_quality} articles below MIN_QUALITY={MIN_QUALITY!r}")

    return chosen


def print_choice_summary(chosen: dict, qualities: dict):
    """How many articles per region, and per quality level."""
    for region in WANTED_REGIONS:
        quality_counts = Counter()
        for title, title_region in chosen.items():
            if title_region == region:
                quality_counts[qualities[title]] += 1

        total = sum(quality_counts.values())
        details = []
        for level in reversed(QUALITY_LEVELS):        # best first
            if quality_counts[level] > 0:
                details.append(f"{level} {quality_counts[level]}")

        print(f"  {region:<12} {total:>4} articles   ({', '.join(details)})")

        if total == 0:
            print(f"  WARNING: nothing found for {region!r}. Check the spelling (it must be an exact title).")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    start = time.time()

    # Pass 1: learn every article's parent and quality.
    print("Pass 1: reading the dump to build the IsPartOf family tree ...")
    parents, qualities, redirects = build_place_tree(DUMP)
    print(f"  {len(qualities):,} articles, {len(parents):,} with a parent, "
          f"{len(redirects):,} redirects   ({time.time() - start:.0f}s)")

    # Decide which articles we want.
    chosen = choose_titles(parents, qualities, redirects)
    print(f"\nChosen: {len(chosen)} articles under {WANTED_REGIONS}")
    print_choice_summary(chosen, qualities)

    # Pass 2: read the dump again and save the chosen articles.
    print("\nPass 2: saving the chosen articles ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for title, text in iter_pages(DUMP):
        if title not in chosen:
            continue

        # Save the raw wikitext. encoding="utf-8" matters on Windows: the
        # articles contain Thai and Bengali script.
        out_path = OUT_DIR / (safe_filename(title) + ".wiki")
        out_path.write_text(text, encoding="utf-8")

        saved.append({"title": title, "chars": len(text)})

    # Write the index of what we saved: WikivoyageLoader reads it.
    # (Old .wiki files of articles no longer chosen may stay in the folder,
    #  but they aren't in the index, so nothing reads them.)
    index_path = OUT_DIR / "_index.json"
    index_json = json.dumps(saved, indent=2)
    index_path.write_text(index_json, encoding="utf-8")

    total_chars = sum(article["chars"] for article in saved)
    print(f"  saved {len(saved)} articles ({total_chars:,} characters of wikitext) to {OUT_DIR}")
    print(f"\nDone in {time.time() - start:.0f}s.")


# This block only runs when you execute the file directly
# (python -m stage2.s1_extract_articles), not when another file imports it.
if __name__ == "__main__":
    main()
