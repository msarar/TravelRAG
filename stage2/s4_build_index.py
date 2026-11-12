"""Stage 2, piece 4: embed every chunk and store it in a Chroma vector database.

WHAT HAPPENS HERE
  For every chunk:
    text  ──embedding model──►  vector (1024 numbers)
  Then Chroma stores, side by side:
    - the vector     (used for searching by meaning)
    - the text       (handed to the LLM later)
    - the metadata   (used for filtering and citing sources)
  ...and saves it all to the stage2/chroma_db/ folder, so we only do this slow work once.

This is the slowest step so far: every chunk goes through the embedding model
on your CPU. The script prints progress and an estimate of the time left.

Run (from the project root):  python -m stage2.s4_build_index
"""

import shutil
import time

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from stage2.chunking import split_into_chunks
from stage2.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_NUM_CTX,
)
from stage2.documents import load_documents

# How many chunks to send to the embedding model at once.
# Bigger batches are a little faster, but you see progress less often.
BATCH_SIZE = 32

# Set to a number (e.g. 100) to index only the first N chunks: useful for a
# quick test run before committing to the full build. None = index everything.
MAX_CHUNKS = None


def clean_metadata(metadata: dict) -> dict:
    """Make metadata Chroma-safe while keeping the SAME keys on every chunk.

    Chroma only accepts str, int, float or bool values: None is rejected.
    Our metadata uses None for "no value", e.g. district=None for "Chiang Mai"
    (it isn't a district of anything).

    We replace None with an empty string "" rather than dropping the key, so
    every chunk has the same set of keys (a uniform "schema"). That means:
      - code can always do metadata["district"] without a KeyError
      - we can still filter on "no district":  {"district": ""}
    The convention to remember: "" means "no value".
    """
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""
        else:
            cleaned[key] = value
    return cleaned


def make_chunk_ids(chunks) -> list[str]:
    """Give every chunk a readable, unique ID: "<title>#<number within article>".

    e.g. "Bangkok/Silom#0", "Bangkok/Silom#1", ... "Dhaka#0", ...

    Chroma requires every ID to be unique. (Using start_index instead would
    NOT be safe: if an article contains the same text twice, two chunks can
    get the same start_index.)
    """
    ids = []
    count_per_title = {}      # how many chunks we've numbered so far, per article

    for chunk in chunks:
        title = chunk.metadata["title"]

        if title not in count_per_title:
            count_per_title[title] = 0

        number = count_per_title[title]
        ids.append(f"{title}#{number}")

        count_per_title[title] += 1

    return ids


def format_seconds(seconds: float) -> str:
    """125.3 -> '2m 05s'"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds:02d}s"


def main():
    # 1. Load and split, exactly as in step 3.
    docs = load_documents()
    chunks = split_into_chunks(docs)

    if MAX_CHUNKS is not None:
        chunks = chunks[:MAX_CHUNKS]
        print(f"TEST RUN: only indexing the first {MAX_CHUNKS} chunks.")

    print(f"{len(docs)} documents -> {len(chunks)} chunks to embed.")

    # 2. Start from an empty database every time.
    #    Otherwise re-running this script would mix old chunks (from older
    #    cleaning/splitting code) with new ones.
    if CHROMA_DIR.exists():
        print(f"Deleting the old index in {CHROMA_DIR}/ ...")
        shutil.rmtree(CHROMA_DIR)

    # 3. Connect to the embedding model (running in Ollama).
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, num_ctx=EMBEDDING_NUM_CTX)

    # 4. Create the vector store.
    #    - embedding_function: Chroma calls this to turn text into vectors
    #    - persist_directory:  where to save; without it, everything lives only
    #                          in memory and is gone when the script ends
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # 5. Prepare the chunks: Chroma-safe metadata (None -> "") and a unique ID for each.
    for chunk in chunks:
        chunk.metadata = clean_metadata(chunk.metadata)

    ids = make_chunk_ids(chunks)

    # 6. Embed and store in batches, so we can show progress.
    #    add_documents() does two things for each batch:
    #      a) calls the embedding model on every chunk's page_content
    #      b) saves vectors + text + metadata into Chroma
    start_time = time.time()
    total = len(chunks)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch_chunks = chunks[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]

        vector_store.add_documents(documents=batch_chunks, ids=batch_ids)

        # Progress and time-remaining estimate.
        done = min(batch_end, total)
        elapsed = time.time() - start_time
        seconds_per_chunk = elapsed / done
        remaining = seconds_per_chunk * (total - done)

        print(f"  embedded {done:>5}/{total}   "
              f"elapsed {format_seconds(elapsed)}   "
              f"about {format_seconds(remaining)} left")

    total_time = time.time() - start_time
    print(f"\nDone: {total} chunks embedded and saved to {CHROMA_DIR}/ "
          f"in {format_seconds(total_time)}.")

    # 7. Quick sanity check: search by meaning.
    #    similarity_search embeds the question with the SAME model, then
    #    returns the k chunks whose vectors are closest to it.
    question = "Where can I eat biryani in Old Dhaka?"
    print(f"\nTest search: {question!r}\n")

    results = vector_store.similarity_search(question, k=3)

    for rank, doc in enumerate(results, start=1):
        title = doc.metadata["title"]
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"{rank}. [{title}] {preview}...\n")


if __name__ == "__main__":
    main()
