"""Shared helper: split Documents into chunks.

Used by s3_split_documents.py (to inspect the chunks) and s4_build_index.py
(to embed and store them), so both always split in exactly the same way.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from stage2.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_into_chunks(docs: list[Document]) -> list[Document]:
    """Split each Document into chunks of at most CHUNK_SIZE characters.

    Every chunk gets a copy of its parent's metadata, plus "start_index":
    the character position where the chunk begins in the original article.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    return chunks
