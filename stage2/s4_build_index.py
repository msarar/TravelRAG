"""Stage 2, piece 4: embed chunks into Chroma, using LangChain's indexing API.

WHAT HAPPENS HERE
  For every chunk:
    text  ──embedding model──►  vector (1024 numbers)
  Chroma stores the vector, the text and the metadata in stage2/chroma_db/.

WHAT'S NEW: THE INDEXING API (instead of "delete everything, embed everything")
  A second, small database, the RECORD MANAGER (stage2/record_manager.db),
  keeps a logbook with one row per chunk:
      key        = hash of the chunk's text + metadata  (also its ID in Chroma)
      group_id   = which article it came from (metadata["source"], the URL)
      updated_at = when a run last saw this chunk

  For every chunk, index() checks the logbook:
      hash already there  -> SKIP (no embedding), just refresh updated_at
      hash is new         -> embed it, add it to Chroma, add a logbook row
  And at the end of each call (cleanup="scoped_full"):
      rows of the articles seen in this call that were NOT seen -> stale -> deleted

  So re-running this script only embeds what changed, and if it crashes,
  the next run carries on where it stopped.

RULE: chroma_db/ and record_manager.db must always be deleted TOGETHER.
  Delete only chroma_db/ and the logbook still says "every chunk exists",
  so everything gets skipped and the index stays empty.
  Use REBUILD = True below instead of deleting by hand.

Run (from the project root):  python -m stage2.s4_build_index
"""

import shutil
import time

from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager
from langchain_core.indexing import index
from langchain_ollama import OllamaEmbeddings

from stage2.chunking import split_one_article
from stage2.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_NUM_CTX,
    RAW_DIR,
    RECORD_MANAGER_DB,
    RECORD_MANAGER_NAMESPACE,
)
from stage2.documents import WikivoyageLoader

# True  = delete chroma_db/ AND record_manager.db, then index everything from zero.
#         Needed once when switching to the indexing API (the old index has
#         "Phuket#3"-style IDs the logbook knows nothing about), and after
#         changing the embedding model.
# False = normal run: only new or changed chunks get embedded.
REBUILD = False

# Set to a number (e.g. 3) to index only the first N articles: a quick test run.
# Safe with scoped_full: articles that aren't processed are left alone.
# None = all articles.
MAX_ARTICLES = None

# How many chunks index() sends to Chroma (and so to the embedding model) at once.
BATCH_SIZE = 32


def clean_metadata(metadata: dict) -> dict:
    """Make metadata Chroma-safe while keeping the SAME keys on every chunk.

    Chroma only accepts str, int, float or bool values: None is rejected.
    We replace None with "" (the convention: "" means "no value").

    With the indexing API this matters twice: the metadata is also part of
    the HASH, so it must look exactly the same on every run.
    """
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""
        else:
            cleaned[key] = value
    return cleaned


def format_seconds(seconds: float) -> str:
    """125.3 -> '2m 05s'"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds:02d}s"


def delete_old_index():
    """Delete the vector index and the logbook TOGETHER (never just one of them)."""
    if CHROMA_DIR.exists():
        print(f"Deleting {CHROMA_DIR}/ ...")
        shutil.rmtree(CHROMA_DIR)

    if RECORD_MANAGER_DB.exists():
        print(f"Deleting {RECORD_MANAGER_DB} ...")
        RECORD_MANAGER_DB.unlink()


def open_record_manager() -> SQLRecordManager:
    """Open (or create) the logbook: a SQLite file with one table, upsertion_record."""
    # SQLAlchemy wants a URL, e.g. "sqlite:///C:/repos/TravelRAG/stage2/record_manager.db".
    # as_posix() turns Windows backslashes into forward slashes.
    db_url = "sqlite:///" + RECORD_MANAGER_DB.as_posix()

    record_manager = SQLRecordManager(
        namespace=RECORD_MANAGER_NAMESPACE,   # which collection these rows belong to
        db_url=db_url,
    )

    # Creates the table if it doesn't exist yet; does nothing if it does.
    record_manager.create_schema()
    return record_manager


def index_one_article(article, record_manager, vector_store) -> dict:
    """Split one article and index its chunks. Returns index()'s counts.

    One index() call per article, so each call sees ALL of that article's
    chunks: scoped_full cleanup then never mistakes "not reached yet" for
    "stale", and every finished article is completely done if we crash later.
    """
    chunks = split_one_article(article)

    for chunk in chunks:
        chunk.metadata = clean_metadata(chunk.metadata)

    result = index(
        chunks,
        record_manager,
        vector_store,
        cleanup="scoped_full",     # delete stale chunks, but only of THIS article
        source_id_key="source",    # metadata["source"] (the URL) -> group_id in the logbook
        key_encoder="sha256",      # hash algorithm (the default sha1 prints a warning)
        batch_size=BATCH_SIZE,
    )
    # result looks like: {"num_added": 3, "num_updated": 0, "num_skipped": 40, "num_deleted": 1}
    result["num_chunks"] = len(chunks)
    return result


def main():
    # 1. Optionally start from zero.
    if REBUILD:
        delete_old_index()

    # 2. Create the loader. This reads NOTHING yet: articles are read one at a
    #    time inside the loop below, via lazy_load() (a generator).
    loader = WikivoyageLoader(RAW_DIR, max_articles=MAX_ARTICLES)

    if MAX_ARTICLES is not None:
        print(f"TEST RUN: only indexing the first {MAX_ARTICLES} articles.")

    # A generator has no len(), so the loader counts the titles in _index.json.
    total_articles = loader.count()
    print(f"{total_articles} articles to index.\n")

    # 3. Connect to the embedding model, the vector store and the logbook.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, num_ctx=EMBEDDING_NUM_CTX)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    record_manager = open_record_manager()

    # 4. Index article by article (streamed), adding up the counts.
    totals = {"num_chunks": 0, "num_added": 0, "num_updated": 0,
              "num_skipped": 0, "num_deleted": 0}
    start_time = time.time()

    # Each pass of this loop asks lazy_load() for the NEXT article: it is read and
    # cleaned right then, indexed, and dropped before the next one is read.
    for article_number, article in enumerate(loader.lazy_load(), start=1):
        title = article.metadata["title"]
        article_start = time.time()

        result = index_one_article(article, record_manager, vector_store)

        for key in totals:
            totals[key] += result[key]

        # Progress: per-article counts, plus a rough estimate by articles done.
        article_seconds = time.time() - article_start
        elapsed = time.time() - start_time
        remaining = elapsed / article_number * (total_articles - article_number)

        print(f"  [{article_number:>3}/{total_articles}] {title:<28} "
              f"chunks {result['num_chunks']:>4}  "
              f"added {result['num_added']:>4}  "
              f"skipped {result['num_skipped']:>4}  "
              f"deleted {result['num_deleted']:>3}   "
              f"{article_seconds:5.1f}s   about {format_seconds(remaining)} left")

    # 5. Summary.
    total_time = time.time() - start_time
    print(f"\nDone in {format_seconds(total_time)}:")
    print(f"  chunks produced : {totals['num_chunks']}")
    print(f"  added (embedded): {totals['num_added']}")
    print(f"  skipped         : {totals['num_skipped']}")
    print(f"  deleted (stale) : {totals['num_deleted']}")

    # Sanity check: Chroma should hold exactly as many chunks as we produced
    # (only true when MAX_ARTICLES is None, i.e. all articles were processed).
    stored = vector_store.get(include=[])     # IDs only, no text or vectors
    print(f"  now in Chroma   : {len(stored['ids'])}")

    # 6. Quick search test, as before.
    question = "Where can I eat biryani in Old Dhaka?"
    print(f"\nTest search: {question!r}\n")

    results = vector_store.similarity_search(question, k=3)

    for rank, doc in enumerate(results, start=1):
        title = doc.metadata["title"]
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"{rank}. [{title}] {preview}...\n")


if __name__ == "__main__":
    main()
