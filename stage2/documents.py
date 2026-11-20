"""Shared helper: load the cleaned Wikivoyage articles as LangChain Documents.

This is a Stage 2 helper module that the step scripts import:
    from stage2.documents import WikivoyageLoader
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
from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# Shared helpers (common/ folder).
from common.paths import safe_filename
from common.wikitext import extract_is_part_of, wikitext_to_text


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


class WikivoyageLoader(BaseLoader):
    """A LangChain document loader for the articles saved by s1_extract_articles.py.

    WHAT IS A BaseLoader?
    LangChain's standard shape for "something that produces Documents".
    A loader only has to write ONE method, lazy_load(), as a generator
    (with yield). BaseLoader then provides, for free:
        load()        -> list(self.lazy_load()): every Document at once, as a list
        alazy_load()  -> an async version (useful later, e.g. in a web API)

    HOW IT'S USED
        loader = WikivoyageLoader(RAW_DIR)      # only stores settings, reads nothing yet
        for doc in loader.lazy_load():          # one article at a time  (s4)
            ...
        docs = loader.load()                    # all articles in a list (s3, explore)
    """

    def __init__(self, raw_dir: Path, max_articles: int | None = None):
        # Just remember the settings. Nothing is read from disk here:
        # the work happens later, when someone iterates lazy_load().
        self.raw_dir = raw_dir
        self.max_articles = max_articles      # None = all articles

    def _read_index(self) -> list[dict]:
        """The article list written by s1, cut to max_articles.

        _index.json looks like [{"title": "Dhaka", "chars": 40549}, ...].
        It only holds titles and sizes, so reading it all at once is cheap.
        """
        index_path = self.raw_dir / "_index.json"
        index_text = index_path.read_text(encoding="utf-8")
        entries = json.loads(index_text)               # string -> list of dicts

        if self.max_articles is not None:
            entries = entries[:self.max_articles]      # slicing the TITLE list is fine

        return entries

    def count(self) -> int:
        """How many articles lazy_load() will yield (our own extra method, for progress output).

        A generator has no len(), so we count the entries in _index.json instead.
        """
        entries = self._read_index()
        return len(entries)

    def lazy_load(self) -> Iterator[Document]:
        """Yield one clean Document per article, reading each file only when it's asked for.

        This is a GENERATOR (it uses yield): every time the caller's for-loop asks
        for the next article, this code runs until the next yield, hands over ONE
        Document, and pauses. So only one article is in memory at a time.
        """
        entries = self._read_index()

        for entry in entries:
            title = entry["title"]

            # 1. Read the raw wikitext from disk.
            raw_path = self.raw_dir / (safe_filename(title) + ".wiki")
            raw_wikitext = raw_path.read_text(encoding="utf-8")

            # 2. Clean it (see common/wikitext.py), and put the title on top as a heading
            #    so the text always says which article it came from.
            clean_text = wikitext_to_text(raw_wikitext)
            page_content = "# " + title + "\n\n" + clean_text

            # 3. Package the text and its metadata into a LangChain Document.
            metadata = build_metadata(title, raw_wikitext)
            doc = Document(page_content=page_content, metadata=metadata)

            # 4. Hand this ONE Document to the caller and pause until they ask for the next.
            yield doc
