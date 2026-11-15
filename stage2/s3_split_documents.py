"""Stage 2, piece 3: split the Documents into section-aware chunks, and check them.

WHY SPLIT?
  - An embedding is ONE vector for a whole piece of text. If you embed the entire
    Bangkok article (170,000 characters) as one vector, it becomes a blurry
    "average" of temples, nightlife, airports and food, and matches nothing well.
  - Small, focused chunks ("street food in Yaowarat") give sharp vectors that
    match specific questions.
  - Whatever we retrieve gets pasted into the LLM's prompt, so chunks must be
    small enough that several of them fit.

HOW? (the code is in chunking.py)
  1. MarkdownHeaderTextSplitter cuts each article at its headings (## Eat, ### Budget...)
  2. RecursiveCharacterTextSplitter cuts sections that are still too long
  3. Every chunk gets a context line on top: "Ao Nang (Krabi Province) > Eat > Budget"

This script also MEASURES the two Stage 1 problems, so we can check they're fixed
before spending ~35 minutes building the index:
  - Stage 1: 49 heading-only chunks (under 60 characters)
  - Stage 1: about half the chunks never mentioned their place name

Run (from the project root):  python -m stage2.s3_split_documents
It also writes stage2/data/chunks.jsonl so you can look through every chunk.
"""

import json

# The splitting itself lives in chunking.py, and the settings (CHUNK_SIZE,
# CHUNK_OVERLAP, CHUNKS_FILE) in config.py, so step 4 can split in exactly the same way.
from stage2.chunking import split_into_chunks
from stage2.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE
from stage2.documents import load_documents

# A chunk whose text (not counting the context line) is shorter than this is
# "tiny". In Stage 1, chunks this small were almost all bare headings.
TINY_CHUNK_CHARS = 60

# Stage 1 numbers, printed next to Stage 2's for comparison.
STAGE1_CHUNKS = 1910
STAGE1_TINY = 49
STAGE1_ORPHAN_PERCENT = 0.52


def chunk_body(chunk) -> str:
    """The chunk's text WITHOUT the context line on top.

    page_content = "Dhaka > Old Dhaka > Drink\nNope. Not in Dhaka."
    body         = "Nope. Not in Dhaka."
    """
    context_line, _newline, body = chunk.page_content.partition("\n")
    return body


def print_length_stats(chunks):
    """Show how long the chunks are, so we can check the splitter did what we asked."""
    lengths = []
    for chunk in chunks:
        lengths.append(len(chunk.page_content))

    average = sum(lengths) / len(lengths)
    over_limit = 0
    for length in lengths:
        if length > CHUNK_SIZE:
            over_limit += 1

    print(f"Chunk length incl. context line (characters): min {min(lengths)}, "
          f"average {average:.0f}, max {max(lengths)}")
    print(f"Chunks longer than CHUNK_SIZE ({CHUNK_SIZE}): {over_limit}")


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


def find_tiny_chunks(chunks):
    """Chunks whose text (without the context line) is very short.

    Stage 1's tiny chunks were bare headings with NO content. Stage 2's should
    be short but real sections, e.g. "Drink: Nope. Not in Dhaka."
    """
    tiny = []
    for chunk in chunks:
        if len(chunk_body(chunk)) < TINY_CHUNK_CHARS:
            tiny.append(chunk)
    return tiny


def count_orphan_chunks(chunks) -> int:
    """How many chunks never mention the place they're about?

    In Stage 1 this was about half. Now every chunk starts with a context line
    containing the place, so this should be 0.
    """
    orphans = 0
    for chunk in chunks:
        place = chunk.metadata["place"]                 # e.g. "Dhaka"
        if place.lower() not in chunk.page_content.lower():
            orphans += 1
    return orphans


def find_chunk_containing(chunks, title, text):
    """The first chunk of article `title` whose text contains `text`, or None."""
    for chunk in chunks:
        if chunk.metadata["title"] == title and text in chunk.page_content:
            return chunk
    return None


def print_chunk(label, chunk, max_chars=600):
    print(f"\n--- {label} ---")
    if chunk is None:
        print("(not found)")
        return
    print("metadata:", chunk.metadata)
    print(chunk.page_content[:max_chars])


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

    # 2. Split (see chunking.py).
    print(f"Splitting with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    chunks = split_into_chunks(docs)
    print(f"Split into {len(chunks)} chunks  (Stage 1: {STAGE1_CHUNKS}).\n")

    # 3. Sizes.
    print_length_stats(chunks)
    print_chunks_per_article(chunks)

    # 4. The two Stage 1 problems, measured again.
    tiny = find_tiny_chunks(chunks)
    orphans = count_orphan_chunks(chunks)
    orphan_percent = orphans / len(chunks)

    print("\nStage 1 problems, re-measured:")
    print(f"  Tiny chunks (< {TINY_CHUNK_CHARS} chars of text): {len(tiny)}   (Stage 1: {STAGE1_TINY}, all bare headings)")
    print(f"  Chunks not mentioning their place:  {orphans} ({orphan_percent:.0%})   (Stage 1: about {STAGE1_ORPHAN_PERCENT:.0%})")

    print("\nFirst 10 tiny chunks (should be short REAL sections, not bare headings):")
    for chunk in tiny[:10]:
        one_line = chunk.page_content.replace("\n", "  |  ")
        print(f"  {one_line}")

    # 5. Two chunks that Stage 1 got wrong.
    biryani = find_chunk_containing(chunks, "Dhaka/Old Dhaka", "Haji Biryani")
    print_chunk("Old Dhaka biryani chunk", biryani)

    visa_list = find_chunk_containing(chunks, "Thailand", "60 days")
    print_chunk("Thailand visa-free list chunk (missed by retrieval in Stage 1)", visa_list)

    # 6. Save everything for you to browse.
    save_chunks(chunks)
    print(f"\nAll chunks written to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
