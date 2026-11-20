# TravelRAG cheat sheet

Every Ollama, LangChain and Chroma call used in the project, with its arguments,
**grouped by the stage where it first appeared**. Each stage lists only what's
*new* in that stage; anything from an earlier stage isn't repeated.
Plain-Python data cleaning (`common/wikitext.py` etc.) is left out on purpose.

**Run any script from the project root:** `python -m stage1.s6_ask` (not `python s6_ask.py`).

**Contents**
- [Stage 0: Setup](#stage-0-setup): Ollama CLI, `OllamaEmbeddings`, `ChatOllama.invoke`
- [Stage 1: Basic RAG](#stage-1-basic-rag): `Document`, text splitter, `Chroma`, `ChatPromptTemplate`, streaming, retriever, LCEL
- [Stage 2: Real ingestion](#stage-2-real-ingestion): `MarkdownHeaderTextSplitter`, two-step splitting with context lines, test runner (`questions.yaml`, `llm.invoke`), `index()` + `SQLRecordManager`, `BaseLoader`, place-aware search

---

## Stage 0: Setup

**New packages:**
```powershell
pip install langchain langchain-ollama numpy
```

### 0.1 Ollama (command line)

| Command | What it does |
|---|---|
| `ollama --version` | Check Ollama is installed |
| `ollama pull <model>` | Download a model, e.g. `ollama pull qwen3:4b-instruct` |
| `ollama list` | Show downloaded models and their sizes |
| `ollama run <model> --verbose "prompt"` | Run one prompt; `--verbose` prints speed stats (`eval rate` = tokens/s) |
| `ollama ps` | Show loaded models; **PROCESSOR** column says CPU or GPU |
| `ollama rm <model>` | Delete a model |

Inside an interactive `ollama run <model>` session:

| Command | What it does |
|---|---|
| `/set parameter num_gpu 0` | Force CPU only |
| `/set nothink` | Turn off "thinking" (for models that support it) |
| `/bye` | Exit |

**Our models:** `qwen3-embedding:0.6b` (embeddings, 1024 dims) and `qwen3:4b-instruct` (chat).
Plain `qwen3:4b` is a *thinking-only* model: it always reasons first and can't be switched off.

### 0.2 Embedding model: `OllamaEmbeddings`  *(s0; later s4, common/vector_store.py)*

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b", num_ctx=2048)
```

| Argument | Meaning |
|---|---|
| `model` | Ollama model name |
| `num_ctx` | Context window in **tokens**: the most text it reads per input. Longer text is **silently cut off**. Small values save memory |

| Method | Input | Returns |
|---|---|---|
| `embed_documents(texts)` | `list[str]` | `list[list[float]]`, one vector per text |
| `embed_query(text)` | `str` | `list[float]`, one vector |

Our vectors are **normalised** (length ≈ 1.0). Check with `np.linalg.norm(vector)`.

### 0.3 Chat model: `ChatOllama`  *(s0; later s6)*

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b-instruct", temperature=0, num_ctx=4096)
```

| Argument | Meaning |
|---|---|
| `model` | Ollama model name |
| `temperature` | `0` = as factual/repeatable as possible; higher = more creative |
| `num_ctx` | Context window in tokens: must fit everything you send **+ the answer** (in Stage 1: instructions + all chunks + question) |

| Method | Input | Returns |
|---|---|---|
| `invoke(input)` | a string, or prompt messages | `AIMessage`; text is in `.content`. Waits for the whole answer |

```python
reply = llm.invoke("What is RAG?")
print(reply.content)
```

---

## Stage 1: Basic RAG

**New packages:**
```powershell
pip install langchain-text-splitters langchain-chroma
```

### 1.1 `Document`  *(stage1/documents.py)*

```python
from langchain_core.documents import Document

doc = Document(page_content="the text", metadata={"title": "Dhaka", "place": "Dhaka"})
```

| Field | Meaning |
|---|---|
| `page_content` | The text. **Only this gets embedded** and shown to the LLM |
| `metadata` | A dict of facts *about* the text. Not embedded; used for filtering and citing |

A Document has no save method: write `doc.page_content` to a file yourself.

### 1.2 Splitting: `RecursiveCharacterTextSplitter`  *(stage1/chunking.py, s3)*

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=True)
chunks = splitter.split_documents(docs)
```

| Argument | Meaning |
|---|---|
| `chunk_size` | Max chunk length, measured in **characters** (by default). 1000 chars ≈ 250 tokens |
| `chunk_overlap` | **Up to** this many characters repeated between neighbouring chunks. Made of whole pieces only, so it can be 0 |
| `add_start_index` | Adds `metadata["start_index"]`: where the chunk starts in the original text |
| `separators` | *(default)* `["\n\n", "\n", " ", ""]`: tries blank lines first, then lines, words, characters |

| Method | Input | Returns |
|---|---|---|
| `split_documents(docs)` | `list[Document]` | `list[Document]`; each chunk gets a **copy** of its parent's metadata |
| `split_text(text)` | `str` | `list[str]` |
| `create_documents(texts, metadatas)` | `list[str]`, `list[dict]` | `list[Document]` |

Metadata does **not** count towards `chunk_size`. Only `page_content` is split.

### 1.3 Vector store: `Chroma`  *(s4, s5, common/vector_store.py)*

#### Create / open
```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="wikivoyage",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)
```

| Argument | Meaning |
|---|---|
| `collection_name` | Name of the "table" inside the database. One embedding model per collection |
| `embedding_function` | Used to embed chunks when adding, and questions when searching. **Must be the same model both times** |
| `persist_directory` | Folder to save to. Without it, everything lives only in memory |

Same call creates a new store or opens an existing one.

#### Add
```python
vector_store.add_documents(documents=chunks, ids=ids)
```

| Argument | Meaning |
|---|---|
| `documents` | `list[Document]`: embeds each `page_content`, stores text + metadata + vector |
| `ids` | `list[str]`, **must be unique**. We use `"<title>#<number>"` |

Rules: metadata values must be `str`, `int`, `float` or `bool`. **No `None`** (we store `""` instead).
The HNSW index is built automatically as you add; there's no separate "create index" step.

#### Search
```python
docs = vector_store.similarity_search("biryani in Old Dhaka", k=4, filter={"place": "Dhaka"})
pairs = vector_store.similarity_search_with_score("biryani in Old Dhaka", k=4, filter={"place": "Dhaka"})
```

| Method | Returns |
|---|---|
| `similarity_search(query, k, filter)` | `list[Document]`, closest first |
| `similarity_search_with_score(query, k, filter)` | `list[(Document, distance)]`, closest first |

| Argument | Meaning |
|---|---|
| `query` | The question (embedded with `embedding_function`) |
| `k` | How many results |
| `filter` | Metadata filter, applied **inside** the database before ranking. `{"place": "Bangkok"}` = exact match, case-sensitive. `None` = no filter |

```python
for doc, distance in pairs:          # unpack each (Document, distance) pair
    print(doc.metadata["title"], distance)
```

#### Read without searching
```python
stored = vector_store.get(include=["metadatas"])
stored["metadatas"]                  # list of metadata dicts
```

| Argument | Meaning |
|---|---|
| `ids` | Fetch these exact IDs |
| `where` | Metadata filter, e.g. `{"place": "Dhaka"}` |
| `include` | Which parts to return: `"documents"`, `"metadatas"`, `"embeddings"` |

`get()` reads straight from SQLite: no embedding, no similarity, no ranking. Without `ids`/`where` it reads **every** record.

#### Reading distances
Chroma's default score is **squared L2 distance**: smaller = better.

| Chroma distance | Cosine | Meaning |
|---|---|---|
| 0.00 | 1.00 | identical meaning |
| ~0.57 | ~0.72 | strong match |
| ~1.0 | ~0.5 | weak |
| 2.00 | 0.00 | unrelated |

For **normalised** vectors only: `cosine = 1 − distance / 2` (helper: `common/distances.py`).

#### What's on disk
```
chroma_db/
├── chroma.sqlite3     text, metadata, IDs, filters (+ a queue of vectors not yet saved to the index)
└── <uuid>/            the HNSW vector index for one collection
```

### 1.4 Prompts: `ChatPromptTemplate`  *(s6)*

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel assistant. Answer ONLY from the excerpts..."),
    ("human", "Travel guide excerpts:\n\n{context}\n\nQuestion: {question}"),
])

messages = prompt.invoke({"context": context, "question": question})
```

| Part | Meaning |
|---|---|
| `("system", ...)` | Instructions the model should follow |
| `("human", ...)` | The user's message |
| `{context}`, `{question}` | Placeholders; filled by the dict keys of the same name |

| Method | Returns |
|---|---|
| `prompt.invoke(dict)` | The filled-in messages, ready for `llm.invoke` / `llm.stream`. Instant: no AI involved |
| `messages.to_string()` | The messages as plain text (handy for printing what the model sees) |

### 1.5 Streaming answers: `ChatOllama.stream`  *(s6)*

| Method | Input | Returns |
|---|---|---|
| `stream(input)` | a string, or prompt messages | an iterator of small chunks; each has `.content`. Pieces arrive as they're generated |

```python
for piece in llm.stream(messages):
    print(piece.content, end="", flush=True)   # end="": no newline per piece; flush=True: show now
```

Same total time as `invoke`; you just start reading straight away.

### 1.6 Retriever  *(s6, LCEL version)*

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 4, "filter": {"place": "Bangkok"}})
docs = retriever.invoke("street food at night")
```

| Argument | Meaning |
|---|---|
| `search_kwargs` | Settings passed to every search: `k`, `filter` |
| `search_type` | *(optional)* `"similarity"` (default), or `"similarity_score_threshold"` with `search_kwargs={"score_threshold": ...}` to drop weak matches |

`retriever.invoke(question)` → `list[Document]` **without distances**. A retriever is the standard "question → Documents" component that fits in a chain.

### 1.7 LCEL: chains with `|`  *(s6)*

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

chain = (
    {
        "context": retriever | RunnableLambda(format_context),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("where can I eat biryani in Old Dhaka?")      # whole answer
for piece in chain.stream("where can I eat biryani in Old Dhaka?"): # piece by piece
    print(piece, end="", flush=True)
```

| Piece | What it does |
|---|---|
| `a \| b` | Run `a`, pass its output to `b` |
| `{ "x": ..., "y": ... }` | Sends the **same input** to every branch, collects results into a dict |
| `RunnablePassthrough()` | Passes its input through unchanged |
| `RunnableLambda(func)` | Wraps a normal Python function so it can sit in a chain |
| `StrOutputParser()` | `AIMessage` → plain text (so you print `piece`, not `piece.content`) |

**Building a chain runs nothing.** Work happens only on `chain.invoke(...)` / `chain.stream(...)`; streaming the chain makes LangChain call `llm.stream` inside.
A chain returns only its **final** output: the retrieved Documents are dropped (so no sources unless you add them).

#### The shared interface ("Runnable")
Prompts, models, retrievers and chains all have the same methods:

| Method | Meaning |
|---|---|
| `.invoke(input)` | Run once, return the full result |
| `.stream(input)` | Run once, yield the result in pieces |

What "run" means depends on the object:

| Object | `.invoke` input → output |
|---|---|
| `prompt` | dict → messages |
| `llm` | messages → `AIMessage` |
| `retriever` | question → `list[Document]` |
| `chain` | first step's input → last step's output |

### 1.8 Order of use in Stage 1

All in `stage1/`; shared helpers in `common/`.

```
s0  OllamaEmbeddings.embed_documents, ChatOllama.invoke            (smoke test)
s2  Document(page_content, metadata)                               (build Documents)
s3  RecursiveCharacterTextSplitter.split_documents                 (chunks)
s4  Chroma(...) → add_documents(documents, ids) → similarity_search (build + test index)
s5  Chroma.get(include) → similarity_search_with_score(query, k, filter)
s6  ChatPromptTemplate.from_messages → prompt.invoke → ChatOllama.stream        (step by step)
    as_retriever → {RunnableLambda, RunnablePassthrough} | prompt | llm | StrOutputParser → chain.stream   (LCEL)
```

---

## Stage 2: Real ingestion

**New packages:**
```powershell
pip install langchain-classic "sqlalchemy[asyncio]"    # SQLRecordManager (2.4); [asyncio] adds greenlet
```
`PyYAML` (for the test runner) already came with LangChain.

### 2.1 Splitting by headings: `MarkdownHeaderTextSplitter`  *(stage2/chunking.py, explore_sections.py)*

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
)
sections = header_splitter.split_text(article.page_content)
```

| Argument | Meaning |
|---|---|
| `headers_to_split_on` | List of `(heading marker, metadata key)`. A new section starts at each of these headings |
| `strip_headers` | *(default `True`)* Removes the heading line from the section's text |

| Method | Input | Returns |
|---|---|---|
| `split_text(text)` | `str` | `list[Document]`, **one per section** |

Each section's metadata holds **only the headings above it**:
```python
{"h1": "Thailand", "h2": "Get in", "h3": "Entry requirements"}
```

Watch out for:
- **No size limit.** A section can be any length, so long sections need splitting again (2.2).
- **Takes a string, not Documents.** The article's metadata (`title`, `place`, `source`...) is
  **not** copied into the sections. Copy it yourself.
- **Headings are stripped from the text.** That's why each chunk gets a context line (2.2).
- **Empty sections are skipped.** A heading followed directly by a sub-heading gives no section of its own.
- Headings must be real Markdown (`## Get in` at the start of a line). `common/wikitext.py` makes them.

### 2.2 Two-step splitting with a context line  *(stage2/chunking.py)*

Step 1 splits by headings, step 2 splits long sections by size, and each chunk starts with
a line saying where it comes from:

```python
for section in sections:
    context_line = "Thailand > Get in > Entry requirements"        # place > section path
    room_for_text = CHUNK_SIZE - len(context_line) - 1             # leave room for the line + "\n"

    piece_splitter = RecursiveCharacterTextSplitter(chunk_size=room_for_text, chunk_overlap=150)
    for piece in piece_splitter.split_text(section.page_content):  # str -> list[str]
        metadata = dict(article.metadata)                          # copy the article's metadata
        metadata.update({"section": "Get in", "subsection": "Entry requirements",
                         "subsubsection": "", "section_path": "Get in > Entry requirements"})
        chunk = Document(page_content=context_line + "\n" + piece, metadata=metadata)
```

| Where it goes | Embedded? | Used for |
|---|---|---|
| Context line, in `page_content` | **Yes** | Every chunk "knows" its place and section, so it can be found by meaning |
| `section`, `section_path`... in `metadata` | No | Showing sources, filtering (e.g. `filter={"section": "Get in"}`) |

`split_text` is used here instead of `split_documents` because we build the Documents
ourselves. So there's no `start_index` in Stage 2. Chunk IDs stay `"<title>#<counter>"`.

### 2.3 Automatic test runs: `questions.yaml` + `s7`  *(stage2/s7_run_questions.py)*

```python
import yaml
questions = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))  # YAML -> list of dicts

results = vector_store.similarity_search_with_score(question, k=TOP_K)   # retrieval, with distances
messages = prompt.invoke({"context": context, "question": question})    # same prompt as s6
reply = llm.invoke(messages)                                             # invoke, not stream
answer = reply.content
```

| Call | Why here |
|---|---|
| `yaml.safe_load(text)` | Reads the question list. `safe_load` (not `load`) won't run code hidden in the file |
| `llm.invoke(messages)` | Returns the whole `AIMessage` at once. No one is watching it type, so no need to `stream` |
| Reusing `prompt`, `format_context`, `TOP_K` from `s6_ask` | The runner tests **exactly** what `s6` does |

Output: `stage2/results/run_<date>_<time>.md` (to read) and `.json` (to compare runs later).

### 2.4 Incremental indexing: `index()` + `SQLRecordManager`  *(s4)*

```powershell
pip install langchain-classic "sqlalchemy[asyncio]"
```

```python
from langchain_classic.indexes import SQLRecordManager
from langchain_core.indexing import index

record_manager = SQLRecordManager(namespace="chroma/wikivoyage",
                                  db_url="sqlite:///" + RECORD_MANAGER_DB.as_posix())
record_manager.create_schema()          # creates the logbook table if missing

result = index(chunks, record_manager, vector_store,
               cleanup="scoped_full", source_id_key="source",
               key_encoder="sha256", batch_size=32)
# {"num_added": 3, "num_updated": 0, "num_skipped": 40, "num_deleted": 1}
```

| Argument | Meaning |
|---|---|
| `namespace` | Label on every logbook row: which collection it belongs to. New embedding model → new collection → new namespace |
| `cleanup` | What to delete: `None` nothing · `"incremental"` per batch · `"full"` everything not seen this run · `"scoped_full"` not-seen chunks of articles seen this run |
| `source_id_key` | Metadata key copied into the logbook as `group_id` (one per article). Required for `incremental`/`scoped_full` |
| `key_encoder` | Hash algorithm. Default `"sha1"` prints a warning; `"sha256"` doesn't |
| `batch_size` | Chunks per logbook check / `add_documents` call |

How it decides, per chunk: **ID = hash(text) + hash(metadata as sorted JSON)**.
Hash in the logbook → skip (just refresh `updated_at`). New → embed, add to Chroma, add a row.
At the end, rows of the seen articles with `updated_at` before the run started → stale → deleted.

The logbook (`record_manager.db`, table `upsertion_record`) has one row per chunk:
`key` (hash), `namespace`, `group_id` (article), `updated_at` (last seen).

Watch out for:
- `index()` ignores your own IDs: the hash IS the ID.
- Delete `chroma_db/` **and** `record_manager.db` together, never one alone.
- The embedding model isn't in the hash: changing it re-embeds nothing unless you change the collection/namespace.
- Interrupting is safe (Ctrl+C): every finished batch of 32 is in Chroma **and** the logbook, so the next run skips it.

**The record manager's methods** (`index()` calls them; you only call `create_schema()`):

| Method | What it does to the logbook table |
|---|---|
| `create_schema()` | creates the table if missing |
| `get_time()` | current time **from the database's clock** (start of a run) |
| `exists(keys)` | which hashes are already there → skip or add |
| `update(keys, group_ids=...)` | insert rows / refresh `updated_at` |
| `list_keys(group_ids=..., before=start)` | rows not seen this run → stale |
| `delete_keys(keys)` | remove those rows |

`InMemoryRecordManager` (in `langchain_core`) has the same methods but forgets everything when the script ends.

#### Recipes: adding to (and changing) the index

**How chunks get in:** never by calling `add_documents` yourself anymore. You change *what s1 extracts*,
then `s4` runs `index()`, which adds only what's new. Always with `REBUILD = False` unless stated.

| I want to… | Change | Run | What `s4` reports |
|---|---|---|---|
| **add articles** (a region) | `WANTED_REGIONS` in `s1`, e.g. `["Thailand", "Bangladesh", "Laos"]` | `s1` → `s4` | old articles `skipped`, new ones `added` |
| **add only better articles** | `MIN_QUALITY = "usable"` in `s1` (or lower it to add outlines) | `s1` → `s4` | new ones `added` (see "remove" below for dropped ones) |
| **pick up Wikivoyage edits** | download a newer dump into `data/` | `s1` → `s4` | changed chunks `added` + old versions `deleted`, the rest `skipped` |
| **resume after Ctrl+C / a crash** | nothing | `s4` | finished work `skipped`, the rest `added` |
| **try it on a few articles first** | `MAX_ARTICLES = 3` in `s4` (back to `None` after) | `s4` | only those 3 processed; nothing else touched |
| **change chunking** (`CHUNK_SIZE`, context line, metadata fields) | `config.py` / `chunking.py` / `documents.py` | (`s3` to inspect) → `s4` | everything `added` + old chunks `deleted`: the text/metadata changed, so every hash did |
| **change the embedding model** | `EMBEDDING_MODEL` **and** `COLLECTION_NAME` in `config.py` | `s4` | everything `added` into the new collection (new namespace); the old one is untouched |
| **start completely over** | `REBUILD = True` in `s4` (back to `False` after) | `s4` | deletes `chroma_db/` + `record_manager.db`, then everything `added` |

**Removing articles** is the one thing that doesn't happen by itself: `scoped_full` only cleans up articles
it *sees* in a run, so an article that's no longer extracted keeps its chunks. Use a run with
`cleanup="full"` over the complete article set (deletes every chunk not seen in that run), or rebuild.

#### Chroma details that matter here

```python
stored = vector_store.get(include=[])     # IDs only: no text, metadata or vectors
count = len(stored["ids"])                # how many chunks are in the collection
```

| Behaviour | Why it matters |
|---|---|
| `add_documents(docs, ids=...)` with an ID that **already exists** overwrites it (LangChain's Chroma does an *upsert*) | a crash between "written to Chroma" and "written to the logbook" can't create duplicates: the re-run writes the same hash ID again |
| `include=[]` / `include=["metadatas"]` in `get()` | read only what you need; `get()` never runs a similarity search |

### 2.5 Document loaders: `BaseLoader`  *(stage2/documents.py)*

```python
from langchain_core.document_loaders import BaseLoader

class WikivoyageLoader(BaseLoader):
    def __init__(self, raw_dir, max_articles=None):   # only stores settings
        ...
    def lazy_load(self):                              # YOU write this: a generator
        for entry in entries:
            ...
            yield doc

loader = WikivoyageLoader(RAW_DIR)
for doc in loader.lazy_load(): ...    # one at a time (s4)
docs = loader.load()                  # all at once, as a list (s3): free from BaseLoader
```

| Method | Who writes it | Returns |
|---|---|---|
| `lazy_load()` | you (with `yield`) | a generator: one `Document` per `next()` |
| `load()` | BaseLoader (don't override) | `list(self.lazy_load())` |
| `alazy_load()`, `aload()` | BaseLoader | async versions |
| `load_and_split()` | BaseLoader | deprecated: split yourself |

Generators: no `len()`, no slicing (`itertools.islice` instead), and they can only be
looped over **once**: a second loop silently gets nothing.

### 2.6 Searching with a place filter: keep the place in the search text  *(s5, s6, s7)*

```python
question, typed_place = parse_input("@cox's bazar: where can I stay?")  # ("where can I stay?", "cox's bazar")
place = resolve_place(typed_place, places)                              # "Cox's Bazar" (exact stored spelling)
search_text = build_search_text(question, place)                        # "Cox's Bazar: where can I stay?"

results = vector_store.similarity_search_with_score(
    search_text,                       # what gets EMBEDDED and compared
    k=4,
    filter={"place": place},           # which chunks are ALLOWED (exact match on metadata)
)
```

The **filter** and the **search text** do different jobs. The filter only narrows *which* chunks can come back;
it does nothing to the vector. Searching `"where can I stay?"` alone gave weak distances (1.31+)
even with the right article. The same `search_text` is also what the LLM sees as the question.

### 2.7 Order of use in Stage 2

Only the changed steps; everything else works as in Stage 1. (Stage 2 has no s2.)

```
s1  pass 1: title -> parent / quality / redirect   pass 2: save articles whose IsPartOf chain reaches WANTED_REGIONS
s3  MarkdownHeaderTextSplitter.split_text → RecursiveCharacterTextSplitter.split_text
    → Document(context_line + "\n" + piece, metadata)                (section-aware chunks)
s4  WikivoyageLoader.count() / lazy_load() → split_one_article → SQLRecordManager.create_schema()
    → index(chunks, record_manager, vector_store, cleanup="scoped_full", ...)   (per article, streamed)
s5  parse_input → resolve_place → build_search_text → similarity_search_with_score(search_text, k, filter)
    (shows metadata["section_path"] instead of start_index)
s6  same search text and filter → prompt.invoke → ChatOllama.stream
s7  yaml.safe_load → (same as s6, with llm.invoke) → report .md + .json   (all test questions)
```

Plain-Python helpers (not LangChain) used above: `parse_input`, `resolve_place`, `build_search_text`
(`common/query_parsing.py`), `split_one_article` (`stage2/chunking.py`).
