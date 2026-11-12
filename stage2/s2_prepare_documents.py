"""Stage 2, piece 2: clean the raw wikitext and inspect the resulting Documents.

The cleaning code lives in common/wikitext.py and the Document building in stage2/documents.py.
This script runs them and writes stage2/data/clean/<title>.md, so you can read exactly
what the embedding model will see.

Nothing later in the pipeline reads stage2/data/clean/: it's only for you to inspect.
You can delete that folder any time; this script recreates it.

Run (from the project root):  python -m stage2.s2_prepare_documents
"""

import json

from common.paths import safe_filename
from stage2.config import CLEAN_DIR, RAW_DIR      # CLEAN_DIR: where the readable copies go
from stage2.documents import load_documents


def count_raw_characters() -> int:
    """Add up the size of all raw articles, using the index from s1_extract_articles.py."""
    index_path = RAW_DIR / "_index.json"
    index_text = index_path.read_text(encoding="utf-8")
    index = json.loads(index_text)      # e.g. [{"title": "Dhaka", "chars": 40549}, ...]

    total = 0
    for entry in index:
        total += entry["chars"]
    return total


def main():
    # 1. Build the Documents. The cleaning happens inside load_documents().
    docs = load_documents()

    # 2. Make sure the output folder exists.
    #    parents=True  -> also create "data" if it's missing
    #    exist_ok=True -> don't complain if the folder is already there
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Write each Document's text to its own .md file.
    clean_chars = 0
    for doc in docs:
        title = doc.metadata["title"]                  # e.g. "Bangkok/Sukhumvit"
        file_name = safe_filename(title) + ".md"       # e.g. "Bangkok__Sukhumvit.md"
        path = CLEAN_DIR / file_name                   # Path objects join with "/"

        text = doc.page_content                        # the cleaned text (a plain string)

        # Path.write_text opens the file, writes the string, and closes it.
        # (Document has no save method: we take the text out and write it ourselves.)
        path.write_text(text, encoding="utf-8")

        clean_chars += len(text)

    # 4. Print a summary so we can see how much the cleaning removed.
    raw_chars = count_raw_characters()
    fraction_kept = clean_chars / raw_chars

    print(f"Built {len(docs)} Documents.")
    print(f"Raw wikitext: {raw_chars:,} chars")
    print(f"Clean text:   {clean_chars:,} chars ({fraction_kept:.0%} of the original)")

    # 5. Show one example so you can see what a Document looks like.
    first_doc = docs[0]
    first_title = first_doc.metadata["title"]

    print("\nExample metadata:")
    print(first_doc.metadata)

    print(f"\nFirst 600 characters of {first_title}:\n")
    print(first_doc.page_content[:600])     # [:600] = the first 600 characters


# Only run main() when this file is executed directly, not when imported.
if __name__ == "__main__":
    main()
