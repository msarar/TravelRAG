# Stage 1 results: basic RAG pipeline

## What changed

First working pipeline, built from scratch:

- Extract 26 articles from the Wikivoyage dump (Dhaka + districts, Bangkok + districts,
  Chiang Mai, Phuket, Phuket Town, Patong, Krabi Town, Ao Nang, Ko Phi Phi, Thailand)
- Clean wikitext into Markdown-style text (`common/wikitext.py`), one LangChain `Document` per article
- Split with `RecursiveCharacterTextSplitter`: **1,000 characters, 150 overlap**
- Embed with `qwen3-embedding:0.6b` (Ollama), store in Chroma
- Retrieve top 4 chunks, answer with `qwen3:4b-instruct` using a "answer ONLY from the excerpts" prompt
- Two versions of the answer step: plain step-by-step Python and an LCEL chain (same answers)

## Numbers

| | |
|---|---|
| Articles | 26 (1.71 M chars raw wikitext → 1.29 M chars clean, 75%) |
| Chunks | 1,910 |
| Index build time | **35m 48s** (~1.1 s per chunk, CPU only) |
| Chat speed | ~8 tokens/s (`qwen3:4b-instruct`, CPU) |
| Embeddings | 1024 dimensions, normalised (length ≈ 1.0) |
| Distances (squared L2) | ~0.57 = strong match, ~1.0 = weak; `cosine = 1 − distance / 2` |

## The good

- **The pipeline works end to end:** question → retrieve → grounded answer with numbered sources.
- **Strong retrieval for specific questions:**
  - "where can I eat biryani in Old Dhaka?" → top 2 are the biryani listings (0.568, 0.701)
  - "cheap hotel near Khao San Road" → top 2 are the budget accommodation chunks (0.568, 0.571)
  - "how do I get from Phuket to Ko Phi Phi?" → correct ferry answer (Rassada Port, 90–120 min, 600/1,000 baht)
- **Grounding works:** "what is the capital of France?" → *"I couldn't find that in the travel guides"* (not "Paris"), with a weak-match warning (best distance 0.971).
- **Weak questions are detectable:** the vague "@Dhaka: how is it?" scored ~1.0 everywhere, so distance alone flags it.
- **Step-by-step and LCEL versions give the same answers.**

## The bad

- **Heading-only chunks: 49 of 1,910** are just a heading (`### Shopping malls`, `### Splurge`).
  Their vague vectors match many questions and push real content out: 3 of 4 results for
  `@Bangkok: street food at night`. Checked all 49:
  - 45: content existed but was separated (heading followed by a block longer than `CHUNK_SIZE`, so the splitter flushed the heading alone)
  - 2: parent headings followed directly by a sub-heading (`## Drink` → `### Coffee`)
  - 1: genuinely empty in the source (`Dhaka/Central` → `## Connect`)
- **Orphan chunks: ~52%** never mention their place name, so their vectors don't know where they are.
  Lists also lost their introductions: the Thailand **visa-free list** chunk started "- 14 days: Myanmar..."
  and never said "visa", so it wasn't retrieved for "do I need a visa for Thailand?".
- **Wrong answers even with decent excerpts** (generation errors):
  - Biryani: merged two restaurants (*Hajjee Beryani*'s address under the name *Haji Biryani*).
  - Visa: read the **visa-on-arrival** list as the **visa-free** list (the real visa-free list wasn't retrieved).
- **Indexing is slow:** ~1.1 s per chunk on CPU. All of Wikivoyage (~35,000 articles, ~440,000 chunks) would take an estimated ~135 hours.

## Still open (for later stages)

- Fix heading-only and orphan chunks → **Stage 2** (section-aware chunking + context lines)
- Score retrieval and generation separately → **Stage 3**
- Generation errors (merged listings, misread lists): try a 7–8B chat model and a stricter prompt → **Stage 3/4**
- Indexing speed / scale → **Stage 2** (incremental indexing) and later
