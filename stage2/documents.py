"""Shared helper: load the cleaned Wikivoyage articles as LangChain Documents.

This is a Stage 2 helper module that the step scripts import:
    from stage2.documents import load_documents
It lives in stage2/ (not common/) because each stage may build its Documents
differently (this stage will add section metadata).

WHAT IS A LANGCHAIN DOCUMENT?
Just two things:
  - page_content: the text that will be embedded and shown to the LLM
  - metadata:     a dict of facts *about* the text (title, place, URL...)

Metadata is NOT embedded, but it stays attached to every chunk cut from the
document. That lets us filter later ("only search Bangkok") and cite sources
("from: Bangkok/Sukhumvit").
"""

import json

from langchain_core.documents import Document

# Shared helpers (common/ folder).
from common.paths import safe_filename
from common.wikitext import extract_is_part_of, wikitext_to_text

# Where s1_extract_articles.py saved the raw articles (set in stage2/config.py).
from stage2.config import RAW_DIR


def build_metadata(title: str, raw_wikitext: str) -> dict:
    """Facts about the article that should travel with every chunk."""

    # str.partition splits at the FIRST "/" and returns 3 parts:
    #   "Bangkok/Silom" -> ("Bangkok", "/", "Silom")
    #   "Chiang Mai"    -> ("Chiang Mai", "", "")      (no "/" found)
    place, _separator, district = title.partition("/")

    if district == "":
        district = None     # clearer than an empty string for "no district"

    # Wikivoyage URLs use underscores instead of spaces.
    url_title = title.replace(" ", "_")
    source_url = "https://en.wikivoyage.org/wiki/" + url_title

    # Where this place sits in the hierarchy, e.g. "Krabi Province" for Ao Nang.
    is_part_of = extract_is_part_of(raw_wikitext)

    metadata = {
        "title": title,
        "place": place,
        "district": district,
        "is_part_of": is_part_of,
        "source": source_url,
    }
    return metadata


def load_documents() -> list[Document]:
    """Read every article saved by s1_extract_articles.py and return clean Documents."""

    # The index file lists which articles were extracted, e.g.
    # [{"title": "Dhaka", "chars": 40549}, {"title": "Bangkok", ...}, ...]
    index_path = RAW_DIR / "_index.json"
    index_text = index_path.read_text(encoding="utf-8")   # file contents as one string
    index = json.loads(index_text)                         # string -> Python list of dicts

    docs = []
    for entry in index:
        title = entry["title"]

        # 1. Read the raw wikitext from disk.
        raw_path = RAW_DIR / (safe_filename(title) + ".wiki")
        raw_wikitext = raw_path.read_text(encoding="utf-8")

        # 2. Clean it (see common/wikitext.py), and put the title on top as a heading
        #    so the text always says which article it came from.
        clean_text = wikitext_to_text(raw_wikitext)
        page_content = "# " + title + "\n\n" + clean_text

        # 3. Package the text and its metadata into a LangChain Document.
        metadata = build_metadata(title, raw_wikitext)
        doc = Document(page_content=page_content, metadata=metadata)

        docs.append(doc)

    return docs
