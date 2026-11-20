# TravelRAG

A hands-on learning project: a retrieval-augmented generation (RAG) system over
Wikivoyage travel guides, built step by step to learn **LLMs, RAG, tool calling
and LangChain**. Everything runs locally: no API keys, no token costs.

Ask a travel question, and the system finds the relevant guide excerpts and has
a local LLM answer from them, with sources:

```
question> how do I get from Phuket to Ko Phi Phi?

You can get from Phuket to Ko Phi Phi by ferry from Rassada Port in Phuket Town,
which takes 90 to 120 minutes. The ferry costs 600 baht one-way or 1,000 baht
round trip when bought at the pier [1].

Sources:
  [1] Phuket  (distance 0.591)  https://en.wikivoyage.org/wiki/Phuket
```

## Stack

| Part | Choice |
|---|---|
| Language | Python 3.12 |
| Framework | LangChain (`langchain`, `langchain-ollama`, `langchain-text-splitters`, `langchain-chroma`) |
| LLM runtime | [Ollama](https://ollama.com), local |
| Chat model | `qwen3:4b-instruct` |
| Embedding model | `qwen3-embedding:0.6b` (1024 dimensions) |
| Vector database | Chroma (saved to disk) |
| Data | [English Wikivoyage dump](https://dumps.wikimedia.org/enwikivoyage/latest/) (CC BY-SA) |

## Project layout

```
TravelRAG/
├── README.md
├── BLUEPRINT.md          roadmap, decisions, findings per stage
├── CHEATSHEET.md         every Ollama / LangChain / Chroma call used, with arguments
├── test_questions.md     test results, Stage 1 vs Stage 2 side by side
├── questions.yaml        the test questions, read by the test runner
├── requirements.txt
├── data/                 the Wikivoyage dump (shared by all stages, not in git)
├── common/               helpers used by every stage
├── stage1/               basic RAG pipeline: its own config, data and index
└── stage2/               section-aware chunking + automatic test runner
```

Each stage has its own folder with its own `config.py`, `data/` and `chroma_db/`,
so stages can be run and compared side by side. Each stage also has a `RESULTS.md`:
what changed, the numbers, and the good and the bad.

## Setup

1. **Install Ollama** from [ollama.com](https://ollama.com), then pull the models:
   ```powershell
   ollama pull qwen3-embedding:0.6b
   ollama pull qwen3:4b-instruct
   ```

2. **Create a virtual environment and install packages** (from the project root):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Download the dump** into `data/` (keep it compressed, ~130 MB):
   ```powershell
   mkdir data
   curl.exe -L -o data\enwikivoyage-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2
   ```

## Running Stage 1

Always run from the **project root**, as modules (`python -m ...`), so scripts can
import from both `common/` and their stage folder:

| Step | Command | What it does |
|---|---|---|
| 0 | `python -m stage1.s0_smoke_test` | Checks the embedding and chat models work |
| 1 | `python -m stage1.s1_extract_articles` | Pulls the chosen articles out of the dump |
| 2 | `python -m stage1.s2_prepare_documents` | Cleans wikitext into LangChain Documents |
| 3 | `python -m stage1.s3_split_documents` | Splits Documents into chunks (for inspection) |
| 4 | `python -m stage1.s4_build_index` | Embeds all chunks into Chroma (**slow: ~35 min on CPU**) |
| 5 | `python -m stage1.s5_search` | Search only: see which chunks come back, with scores |
| 6 | `python -m stage1.s6_ask` | Full RAG: retrieve, then the LLM answers with sources |

Steps 1 and 4 only need to run once (or after changing the data, cleaning or chunking).
In steps 5 and 6, filter by place with `@bangkok: street food at night`.

Stage 1 covers: Dhaka (+ districts), Bangkok (+ districts), Chiang Mai, Phuket,
Phuket Town, Patong, Krabi Town, Ao Nang, Ko Phi Phi and Thailand: 26 articles,
1,910 chunks. Settings are in `stage1/config.py`.

## Running Stage 2

Same idea as Stage 1, with `stage2` instead of `stage1`. The pipeline itself is
just **s1 → s4**; the other scripts are for inspecting, searching and testing:

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m stage2.s1_extract_articles` | Pulls the chosen articles out of the dump |
| 3 | `python -m stage2.s3_split_documents` | *Optional:* shows the chunks and their stats, writes `chunks.jsonl` |
| 4 | `python -m stage2.s4_build_index` | Embeds into Chroma with LangChain's indexing API: only new or changed chunks are embedded, and it resumes after a crash |
| 5 | `python -m stage2.s5_search` | Search only |
| 6 | `python -m stage2.s6_ask` | Full RAG with sources |
| 7 | `python -m stage2.s7_run_questions` | Runs every question in `questions.yaml`, writes a report to `stage2/results/` (~10 min) |
| – | `python -m stage2.explore_sections` | Shows how one article splits into sections |

There is no `s2` in Stage 2: it only wrote cleaned copies for reading, which
`s3`'s `chunks.jsonl` now covers. In `s4`, set `REBUILD = True` only to start
from zero (it deletes `chroma_db/` and `record_manager.db` together).

What's new: articles are split **by section first** (`MarkdownHeaderTextSplitter`), and
every chunk starts with a context line like `Thailand > Get in > Entry requirements`.
`CHUNK_SIZE` is 2000. Stage 2 covers 47 articles (the Stage 1 set plus 21 more in Bangladesh): 1,623 chunks. Settings are in `stage2/config.py`;
results are in `stage2/RESULTS.md`.

## Status

- [x] **Stage 0**: setup
- [x] **Stage 1**: basic RAG pipeline
- [ ] **Stage 2**: real ingestion: section-aware chunking ✓, test runner ✓; scaling to more of Wikivoyage in progress
- [ ] **Stage 3**: evaluation
- [ ] **Stage 4**: better retrieval (hybrid search, reranking)
- [ ] **Stage 5**: tool calling with a SQLite database
- [ ] **Stage 6**: agent (LangGraph)
- [ ] **Stage 7**: production (API, UI, tracing, Docker)

Stage 2 fixed Stage 1's chunking problems (heading-only chunks, chunks that lose
their place name). Remaining issues (ranking within an article, small-model answer
slips) are documented in `stage2/RESULTS.md`, `BLUEPRINT.md` and `test_questions.md`.

## Notes

- Answers come from Wikivoyage text and a small local model, so they can be
  wrong or out of date. Check official sources for anything important (visas especially).
- Wikivoyage content is licensed CC BY-SA 3.0.
