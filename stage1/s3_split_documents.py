"""Stage 1, piece 3: split the Documents into chunks.

WHY SPLIT?
  - An embedding is ONE vector for a whole piece of text. If you embed the entire
    Bangkok article (170,000 characters) as one vector, it becomes a blurry
    "average" of temples, nightlife, airports and food, and matches nothing well.
  - Small, focused chunks ("street food in Yaowarat") give sharp vectors that
    match specific questions.
  - Whatever we retrieve gets pasted into the LLM's prompt, so chunks must be
    small enough that several of them fit.

HOW? (the code is in chunking.py)
LangChain's RecursiveCharacterTextSplitter tries to cut at the most natural
boundary first, and only falls back to rougher cuts when a piece is still too big:
    1. blank lines ("\n\n")  -> between paragraphs
    2. single newlines ("\n") -> between list items (our restaurant listings!)
    3. spaces (" ")           -> between words
    4. "" (anywhere)          -> last resort, mid-word

Run (from the project root):  python -m stage1.s3_split_documents
It also writes stage1/data/chunks.jsonl so you can look through every chunk.
"""

import json

# The splitting itself lives in chunking.py, and the settings (CHUNK_SIZE,
# CHUNK_OVERLAP, CHUNKS_FILE) in config.py, so step 4 can split in exactly the same way.
from stage1.chunking import split_into_chunks
from stage1.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE
from stage1.documents import load_documents


def print_length_stats(chunks):
    """Show how long the chunks are, so we can check the splitter did what we asked."""
    lengths = []
    for chunk in chunks:
        lengths.append(len(chunk.page_content))

    average = sum(lengths) / len(lengths)

    print(f"Chunk length (characters): min {min(lengths)}, "
          f"average {average:.0f}, max {max(lengths)}")


def print_chunks_per_article(chunks):
    """Count how many chunks each article was cut into."""
    counts = {}
    for chunk in chunks:
        title = chunk.metadata["title"]
        if title not in counts:
            counts[title] = 0
        counts[title] += 1

    # Sort so the articles with the most chunks come first.
    sorted_titles = sorted(counts, key=counts.get, reverse=True)

    print("\nChunks per article (top 8):")
    for title in sorted_titles[:8]:
        print(f"  {title:<32} {counts[title]:>4}")


def count_orphan_chunks(chunks):
    """How many chunks never mention the place they're about?

    Example: a chunk that is just "- Hotel Star — Taher Bagh Lane. Fantastic biryani."
    The embedding of that text has no idea it's in Dhaka, so the question
    "biryani in Dhaka" may not find it. Keep this number in mind: we fix it in Stage 2.
    """
    orphans = 0
    example = None

    for chunk in chunks:
        place = chunk.metadata["place"]                 # e.g. "Dhaka"
        text = chunk.page_content

        if place.lower() not in text.lower():
            orphans += 1
            if example is None:
                example = chunk

    percent = orphans / len(chunks)
    print(f"\nChunks that never mention their place name: {orphans} of {len(chunks)} ({percent:.0%})")
    return example


def save_chunks(chunks):
    """Write every chunk to a JSON Lines file: one JSON object per line."""
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for number, chunk in enumerate(chunks):
            record = {
                "chunk": number,
                "metadata": chunk.metadata,
                "text": chunk.page_content,
            }
            line = json.dumps(record, ensure_ascii=False)   # keep Thai/Bengali readable
            f.write(line + "\n")


def main():
    # 1. Load the 26 cleaned Documents (from documents.py).
    docs = load_documents()
    print(f"Loaded {len(docs)} Documents.")

    # 2. Split (see chunking.py). Each chunk is a NEW Document that gets a
    #    COPY of its parent's metadata, plus "start_index".
    print(f"Splitting with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    chunks = split_into_chunks(docs)
    print(f"Split into {len(chunks)} chunks.\n")

    # 3. Look at the result.
    print_length_stats(chunks)
    print_chunks_per_article(chunks)
    orphan_example = count_orphan_chunks(chunks)

    # 4. Show one chunk in full, with its metadata.
    sample = chunks[5]
    print("\n--- Example chunk ---")
    print("metadata:", sample.metadata)
    print(sample.page_content)

    if orphan_example is not None:
        print("\n--- Example chunk that doesn't mention its place ---")
        print("metadata:", orphan_example.metadata)
        print(orphan_example.page_content[:500])

    # 5. Save everything for you to browse.
    save_chunks(chunks)
    print(f"\nAll chunks written to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
