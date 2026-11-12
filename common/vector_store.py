"""Shared helper: open a Chroma vector store that a build step already saved.

Every setting is passed in as an argument (instead of read from a config file),
so each stage can open its OWN index with its OWN settings.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def open_vector_store(
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    embedding_num_ctx: int,
) -> Chroma:
    """Open an existing index. Fails clearly if it hasn't been built yet."""
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"No index found in {chroma_dir}. Run that stage's s4_build_index first."
        )

    # We still need the embedding model: every QUESTION has to be turned into
    # a vector before we can search. It MUST be the same model that embedded
    # the chunks, or the question vector and chunk vectors aren't comparable.
    embeddings = OllamaEmbeddings(model=embedding_model, num_ctx=embedding_num_ctx)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    return vector_store
