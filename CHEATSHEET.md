# TravelRAG cheat sheet

Every Ollama, LangChain and Chroma call used in the project, with its arguments,
**grouped by the stage where it first appeared**. Each stage lists only what's
*new* in that stage; anything from an earlier stage isn't repeated.
Plain-Python data cleaning (`common/wikitext.py` etc.) is left out on purpose.

**Run any script from the project root:** `python -m stage1.s6_ask` (not `python s6_ask.py`).

**Contents**
- [Stage 0: Setup](#stage-0-setup): Ollama CLI, `OllamaEmbeddings`, `ChatOllama.invoke`
- [Stage 1: Basic RAG](#stage-1-basic-rag): `Document`, text splitter, `Chroma`, `ChatPromptTemplate`, streaming, retriever, LCEL
- [Stage 2: Real ingestion](#stage-2-real-ingestion): *(to come)*

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

*(to come: only what's new in Stage 2 goes here)*
