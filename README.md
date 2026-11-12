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
├── test_questions.md     test questions and results, re-run after each change
├── requirements.txt
├── data/                 the Wikivoyage dump (shared by all stages, not in git)
├── common/               helpers used by every stage
└── stage1/               basic RAG pipeline: its own config, data and index
```

Each stage has its own folder with its own `config.py`, `data/` and `chroma_db/`,
so stages can be run and compared side by side.

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

## Status

- [x] **Stage 0**: setup
- [x] **Stage 1**: basic RAG pipeline
- [ ] **Stage 2**: real ingestion (section-aware chunking, more of Wikivoyage)
- [ ] **Stage 3**: evaluation
- [ ] **Stage 4**: better retrieval (hybrid search, reranking)
- [ ] **Stage 5**: tool calling with a SQLite database
- [ ] **Stage 6**: agent (LangGraph)
- [ ] **Stage 7**: production (API, UI, tracing, Docker)

Known Stage 1 issues (heading-only chunks, chunks that lose their place name, LLM
answer errors) are documented in `BLUEPRINT.md` and `test_questions.md`.

## Notes

- Answers come from Wikivoyage text and a small local model, so they can be
  wrong or out of date. Check official sources for anything important (visas especially).
- Wikivoyage content is licensed CC BY-SA 3.0.
