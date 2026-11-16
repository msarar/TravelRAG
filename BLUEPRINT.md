# TravelRAG — Project Blueprint

A hands-on learning project: build a production-grade RAG system over travel guides, step by step, to learn **LLMs, RAG, tool calling and LangChain**.

## Decisions so far

| Area | Choice | Why |
|---|---|---|
| Language | Python | |
| Corpus | English Wikivoyage dump (`enwikivoyage-latest-pages-articles.xml.bz2` from `https://dumps.wikimedia.org/enwikivoyage/latest/`) | Free (CC BY-SA), consistent section structure (Get in / See / Eat / Sleep…), templated listings usable as structured data |
| LLM runtime | Ollama (local) | No token costs, first-class LangChain integration (`langchain-ollama`), OpenAI-compatible server |
| Chat model | `qwen3:4b-instruct` (Qwen3-4B-Instruct-2507; try a 7–8B model for agent stages) | Non-thinking model: direct answers, supports tool calling. Plain `qwen3:4b` is the thinking-only variant and always reasons first (much slower) |
| Embeddings | `qwen3-embedding:0.6b` via `OllamaEmbeddings`, 1024 dims, `num_ctx=2048` | Strong open model, fully local, nothing sent to the cloud |
| Vector store | Chroma (local) | Simple, persistent, LangChain-native |
| Structured data | SQLite (Wikivoyage listings, OpenFlights routes) | Backs the tool-calling layer |

**Hardware:** i7-1185G7, 16 GB RAM, Intel Iris Xe (no discrete GPU).

**Rules to remember**
- Index and query with the *same* embedding model; changing it means re-embedding everything.
- Dimensions ≠ quality. Chunking, hybrid search and reranking matter more.

---

## Project layout

```
TravelRAG/
├── BLUEPRINT.md, CHEATSHEET.md, test_questions.md   ← docs for all stages
├── data/                  ← shared: the Wikivoyage dump only
├── common/                ← helpers used by every stage (settings passed in as arguments)
│   ├── wikitext.py           wikitext → clean text
│   ├── query_parsing.py      "@place: question" parsing, place list
│   ├── vector_store.py       open_vector_store(chroma_dir, collection, model, num_ctx)
│   ├── distances.py          distance_to_cosine()
│   └── paths.py              safe_filename()
└── stage1/                ← one folder per stage; each has its own settings, data and index
    ├── config.py             models, chunk sizes, and ALL stage-1 paths
    ├── documents.py, chunking.py
    ├── s0_smoke_test.py … s6_ask.py
    ├── data/                 raw/, clean/, chunks.jsonl
    └── chroma_db/            the stage-1 vector index
```

**Run scripts from the project root, as modules:**
```powershell
cd C:\repos\TravelRAG
python -m stage1.s6_ask
```
(`-m` lets scripts import from both `common/` and their own stage folder.)

Stages are **not frozen**: a fix in `common/` applies to every stage. A new stage starts as a copy of the previous stage's folder, with its own `config.py` paths.

---

## Roadmap

| Stage | What you build | What you learn |
|---|---|---|
| 0. Setup | Ollama, models, Python env, Wikivoyage dump | Local LLMs, embeddings basics |
| 1. Basic RAG | Q&A over ~5 cities | The core RAG loop, LangChain fundamentals |
| 2. Real ingestion | Full Wikivoyage, cleaned and chunked well | Data cleaning, chunking, metadata, persistence |
| 3. Evaluation | A test set and scores | How to *measure* RAG quality |
| 4. Better retrieval | Hybrid search, reranking, query rewriting | Production RAG techniques |
| 5. Tools + database | SQLite of places and flights; tools the LLM can call | Tool calling, structured data |
| 6. Agent | One assistant that decides when to search, query or combine | Agents with LangGraph, memory, multi-step reasoning |
| 7. Production | API, UI, tracing, guardrails, Docker | Making it reliable and deployable |

### Stage 0 — Setup
- [x] Install Ollama
- [x] `ollama pull qwen3-embedding:0.6b` and `ollama pull qwen3:4b-instruct`
- [x] Measure tokens/s (`ollama run qwen3:4b-instruct --verbose`) and check `ollama ps` → runs 100% on CPU, ~8 tokens/s
- [x] Python venv + `pip install langchain langchain-ollama numpy` (LangChain 1.4, langchain-ollama 1.1, LangGraph included)
- [x] Download the Wikivoyage dump into `data/` (keep it compressed) — 133 MB
- [x] Smoke test (`s0_smoke_test.py`): 1024 dims; food vs food 0.712, food vs visa 0.416; chat model answers

### Stage 1 — Basic RAG
- [x] Extract articles from the dump (`s1_extract_articles.py`): Dhaka + districts, Bangkok + districts, Chiang Mai, Phuket, Phuket Town, Patong, Krabi Town, Ao Nang, Ko Phi Phi, Thailand (26 articles)
- [x] Clean wikitext into LangChain Documents (`wikitext.py`, `documents.py`, `s2_prepare_documents.py`)
- [x] Split into chunks (`chunking.py`, `s3_split_documents.py`): 1000 chars, 150 overlap → 1,910 chunks
- [x] Embed + store in Chroma (`s4_build_index.py`): 1,910 chunks in 35m 48s (~1.1 s/chunk on CPU)
- [x] Retrieve with scores and place filters (`s5_search.py`, `store.py`, `query_parsing.py`)
- [x] Answer with the LLM (`s6_ask.py`): step-by-step version and LCEL version (both give the same answers)
- [x] Concepts so far: `Document`, text splitters, `OllamaEmbeddings`, Chroma vector store, similarity search, metadata filters
- [x] Concepts: retrievers, `ChatPromptTemplate`, streaming, LCEL (`|` pipes, `RunnablePassthrough`, `RunnableLambda`, `StrOutputParser`)
- **Done when:** "Where can I eat biryani in Old Dhaka?" is answered correctly from the guide, with sources. ✅ Pipeline complete; answer quality issues logged below.

### Stage 1 results → full details in [`stage1/RESULTS.md`](stage1/RESULTS.md)

**Setup:** 26 articles → 1,910 chunks (1,000 chars, 150 overlap) · index built in 35m 48s (~1.1 s/chunk, CPU) · chat ~8 tokens/s

**Good**
- **Pipeline works end to end:** question → top 4 chunks → grounded answer with numbered sources; step-by-step and LCEL versions give the same answers.
- **Specific questions retrieve well:** "biryani in Old Dhaka" → the biryani listings (distance 0.568); "cheap hotel near Khao San Road" → budget accommodation (0.568).
- **Correct answer when retrieval is right:** Phuket → Ko Phi Phi ferry from Rassada Port, 90–120 min, 600/1,000 baht.
- **Grounding works:** "what is the capital of France?" → *"I couldn't find that in the travel guides"*, with a weak-match warning (0.971).
- **Distances are meaningful:** embeddings are normalised, so `cosine = 1 − distance / 2`; ~0.5 strong, ~1.0 weak.

**Bad**
- **49 heading-only chunks** (`### Shopping malls`, `### Splurge`): 45 separated from their content by the splitter, 2 parent headings, 1 empty in the source. They took 3 of 4 slots for `@bangkok: street food at night`.
- **~52% orphan chunks:** never name their place, so their vectors don't know where they are.
- **Lists lost their introduction:** the Thailand visa-free list chunk started "- 14 days: Myanmar…", never said "visa", and wasn't retrieved.
- **Wrong answers from the LLM:** merged two restaurants (*Hajjee Beryani*'s address under *Haji Biryani*); read the visa-on-arrival list as visa-free.
- **Slow indexing:** ~1.1 s/chunk; all of Wikivoyage (~35,000 articles, ~440,000 chunks) ≈ 135 hours.

### Stage 2 — Real ingestion
- [ ] Parse all of Wikivoyage (wikitext → clean text; `mwxml` / `mwparserfromhell`)
- [x] Section-aware chunking with metadata (`MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter`): keeps every heading attached to its content
- [x] Context line like `Dhaka > Old Dhaka > Eat` at the start of every chunk (fixes orphan chunks); `CHUNK_SIZE` 2,000
- [x] Rebuild index (1,211 chunks, 39m 13s) and re-run test questions (14 questions, automatic runner `s7_run_questions.py`)
- [ ] Decide how much of Wikivoyage to index, given ~1.1 s/chunk on CPU
- [ ] Metadata filtering (e.g. only *Eat* sections in Portugal)
- [ ] Persist the index; incremental updates via LangChain's indexing API
- **Done when:** the whole corpus is indexed and any destination is queryable.

### Stage 2 results → full details in [`stage2/RESULTS.md`](stage2/RESULTS.md)

**Setup:** same 26 articles → 1,211 chunks (section-aware, context line on every chunk, 2,000 chars incl. context line) · index built in 39m 13s

**Good**
- **Heading-only chunks fixed:** 49 → 0. Only 6 chunks are under 60 characters of text, and all are real content (e.g. `Old Dhaka > Drink: Nope. Not in Dhaka.`).
- **Orphan chunks fixed:** ~52% → 0%. Every chunk starts with e.g. `Thailand (Southeast Asia) > Get in > Entry requirements`.
- **Visa question fixed at retrieval:** the visa-free list (now in the same chunk as "may enter Thailand for tourism visa-free…") is #1 (0.722), visa-on-arrival rules #2.
- **Better scores on specific questions:** biryani 0.568 → 0.480; "medical clinics in Chiang Mai" → `Cope > Medical` (0.485).
- **Only `chunking.py` changed** (plus settings); every chunk now also has `section` metadata for filtering.

**Bad**
- **Bigger chunks can blur topics:** Yaowarat (Stage 1's best Bangkok street-food result) dropped out of the top 4; scores weaker (0.89–1.02).
- **Same-section chunks look alike:** for Phuket → Ko Phi Phi, `Get in > By car` (0.533) outranks the ferry chunk `Get in > By boat` (0.601).
- **Tiny chunks attract vague questions:** `Drink: Nope. Not in Dhaka.` is #1 for "@dhaka: how is it?" (correctly weak: 1.29–1.40).
- **Near-duplicates across articles:** the same Dhakaiya Biryani paragraph in `Dhaka` and `Dhaka/Old Dhaka` takes two result slots.
- **Indexing speed unchanged:** 39m 13s for about the same amount of text.
- **The 4B model still makes generation errors:** merged two neighbouring hotel listings (Khao San), muddled ferry prices, one claim contradicting its source (vaping), swapped citations.

**Answers (14 questions, checked by hand): 10 ✅ · 4 ⚠️ · 0 ❌.** Both Stage 1 answer failures are fixed: the biryani places are no longer merged, and the visa answer separates visa-free from visa-on-arrival.

### Stage 3 — Evaluation (before optimizing)
- [ ] Write 30–50 test questions with known answers
- [ ] Measure retrieval (right chunk in top k?) and answer quality (correct, grounded), using the local model as judge
- [ ] Score retrieval and generation **separately**: Stage 1 showed both can fail on their own (merged restaurants, misread visa list)
- **Why:** every later change gets a score, so you know whether it actually helped.

### Stage 4 — Better retrieval (measure each against Stage 3)
- [ ] Hybrid search: BM25 + vectors (`EnsembleRetriever`)
- [ ] Reranking with a cross-encoder (e.g. Qwen3-Reranker, bge-reranker)
- [ ] Query rewriting / multi-query
- [ ] Parent-document retrieval
- [ ] Generation fixes, measured on the Stage 1 failures: a 7–8B chat model; a stricter prompt ("copy names, addresses and prices exactly; never combine two listings")

### Stage 5 — Tools and structured data
- [ ] Extract `{{see}}`, `{{eat}}`, `{{sleep}}` listings into SQLite; add OpenFlights routes
- [ ] Tools with `@tool`: `search_places(city, type, price)`, `find_direct_flights(from, to)`
- [ ] Wire up with `bind_tools()`; inspect tool calls and execute them
- [ ] Compare safe fixed-query tools vs text-to-SQL, and their risks

### Stage 6 — The agent
- [ ] Retriever as a tool: *search guides*, *search places*, *find flights*
- [ ] Build with `create_agent` (LangGraph), then a custom LangGraph graph
- [ ] Conversation memory ("what about hotels there?")
- **Done when:** "Plan 3 days in Istanbul with cheap food near Sultanahmet, and can I fly direct from Dhaka?" gets one coherent answer.

### Stage 7 — Production-grade
- [ ] FastAPI backend with streaming; Streamlit or Gradio UI
- [ ] Source citations (article + section)
- [ ] Guardrails: "I don't know" on weak retrieval; prompt-injection defence
- [ ] Observability: Langfuse (self-hosted) or LangSmith
- [ ] Caching, config, tests, Docker

### Optional extras
- [ ] French Wikivoyage (multilingual retrieval)
- [ ] Live travel advisories (data freshness)
- [ ] Swap in a bigger or cloud LLM with a one-line change

---

**Rough pace:** Stages 1–3 are the foundation (a week or two of evenings). Stages 4–6 are the interesting part; Stage 7 is polish.
