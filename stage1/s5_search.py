"""Stage 1, piece 5: retrieval. Ask questions, see which chunks come back.

No LLM yet: this is only the "R" in RAG. Before connecting a language model,
it's worth checking that the RIGHT chunks are being found, because the LLM can
only answer well from what retrieval gives it.

HOW TO USE (type at the prompt):
    where can I get street food in Bangkok?        -> search everything
    @chiang mai: where can I get a massage?        -> only chunks whose place is "Chiang Mai"
                                                      (any upper/lower case works)
    quit                                           -> exit

Run (from the project root):  python -m stage1.s5_search
"""

from common.distances import distance_to_cosine
from common.query_parsing import list_places, parse_input, resolve_place
from common.vector_store import open_vector_store
from stage1.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_NUM_CTX

# How many chunks to return for each question.
TOP_K = 4

# How much of each chunk to print.
PREVIEW_CHARS = 300


def print_results(results):
    """Show each returned chunk with its score, source and a preview."""
    if not results:
        print("  No results.")
        return

    for rank, (doc, distance) in enumerate(results, start=1):
        cosine = distance_to_cosine(distance)
        title = doc.metadata["title"]
        start_index = doc.metadata.get("start_index", "?")

        preview = doc.page_content[:PREVIEW_CHARS]
        preview = preview.replace("\n", " ")             # one line, easier to read

        print(f"\n  {rank}. [{title}]  distance {distance:.3f}  (cosine ≈ {cosine:.3f})  start_index {start_index}")
        print(f"     {preview}...")


def main():
    print("Opening the index...")
    vector_store = open_vector_store(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_num_ctx=EMBEDDING_NUM_CTX,
    )

    places = list_places(vector_store)
    print("Places you can filter on:", ", ".join(places))
    print("Type a question, '@place: question' to filter, or 'quit'.\n")

    while True:
        user_input = input("question> ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break
        if not user_input.strip():
            continue

        question, typed_place = parse_input(user_input)

        # Build the metadata filter (or no filter).
        # {"place": "Bangkok"} means: only chunks where metadata["place"] == "Bangkok"
        search_filter = None
        if typed_place is not None:
            place = resolve_place(typed_place, places)     # "bangkok" -> "Bangkok"

            if place is None:
                print(f"  Unknown place {typed_place!r}. Choose from: {', '.join(places)}\n")
                continue

            search_filter = {"place": place}
            print(f"  (only searching chunks where place == {place!r})")

        # similarity_search_with_score:
        #   1. embeds the question with the same model as the chunks
        #   2. finds the TOP_K closest chunk vectors (HNSW index)
        #   3. returns (Document, distance) pairs, closest first
        results = vector_store.similarity_search_with_score(
            question,
            k=TOP_K,
            filter=search_filter,
        )

        print_results(results)
        print()


if __name__ == "__main__":
    main()
