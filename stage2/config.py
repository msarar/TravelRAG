"""Stage 2 settings, in one place.

Every Stage 2 script reads its models, chunk sizes and folders from here.
Each stage has its own config.py, so stages never overwrite each other's
data or index.
"""

from pathlib import Path

# --- Folders ---------------------------------------------------------------
# Path(__file__) is this file (stage2/config.py). .resolve() makes it a full
# path, and .parent goes up one folder. Building paths from this file's location
# (instead of "data/raw") means they're correct no matter which folder you run from.
STAGE_DIR = Path(__file__).resolve().parent         # .../TravelRAG/stage2
PROJECT_ROOT = STAGE_DIR.parent                     # .../TravelRAG

# The downloaded Wikivoyage dump is shared by all stages.
DUMP_FILE = PROJECT_ROOT / "data" / "enwikivoyage-latest-pages-articles.xml.bz2"

# Everything Stage 2 produces lives inside stage2/.
DATA_DIR = STAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                 # s1: raw wikitext per article
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"    # s3: every chunk, for reading only
CHROMA_DIR = STAGE_DIR / "chroma_db"       # s4: the vector index
RECORD_MANAGER_DB = STAGE_DIR / "record_manager.db"   # s4: the indexing logbook (hash, article, last seen)

# --- Ollama models ---------------------------------------------------------
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:4b-instruct"

# Context window for the embedding model, in tokens. Our chunks are at most
# 2000 characters ≈ 500 tokens (more for Thai/Bengali script), so 2048 is
# enough, and a small window uses much less memory.
EMBEDDING_NUM_CTX = 2048

# Context window for the CHAT model, in tokens. It must hold everything we send
# (instructions + all retrieved chunks + the question) AND the answer it writes.
# 4 chunks x up to ~500 tokens + ~200 for instructions and question ≈ 2,200 tokens,
# plus the answer. 6144 leaves comfortable room (4096 would be tight).
CHAT_NUM_CTX = 6144

# --- Chunking --------------------------------------------------------------
# Measured in CHARACTERS, and includes the context line on top of each chunk.
# Rough rule for English: 1 token ≈ 4 characters, so 2000 characters ≈ 500 tokens.
# Stage 1 used 1000. Stage 2 uses 2000 so a list stays in the same chunk as the
# sentence that explains it (e.g. Thailand's "visa-free for up to..." + the country list).
CHUNK_SIZE = 2000

# Up to this many characters can be repeated between neighbouring chunks
# (only whole pieces, as we saw in step 3).
CHUNK_OVERLAP = 150

# --- Vector database -------------------------------------------------------
# A "collection" is like a sheet in an Excel file: one per embedding model.
COLLECTION_NAME = "wikivoyage"

# --- Indexing logbook (record manager) --------------------------------------
# Labels every logbook row with the collection it belongs to, so one logbook
# could track several collections without mixing them up.
# New embedding model -> new COLLECTION_NAME -> new namespace automatically.
RECORD_MANAGER_NAMESPACE = "chroma/" + COLLECTION_NAME
