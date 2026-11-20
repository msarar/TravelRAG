# Stage 2 results: section-aware chunking

Same 26 articles as Stage 1, so any difference comes from chunking alone.

## What changed

Only `chunking.py` (plus settings and display):

1. **Split by section first** with `MarkdownHeaderTextSplitter` (at `#`, `##`, `###`, `####`).
   Headings move into metadata; sections with no text of their own produce no chunk.
2. **Split long sections** with `RecursiveCharacterTextSplitter`, leaving room for the context line.
3. **Context line on every chunk**, built from the metadata, e.g.
   `Thailand (Southeast Asia) > Get in > Entry requirements`. Includes `is_part_of` when the title doesn't already say it.
4. **Uniform section metadata** on every chunk: `section`, `subsection`, `subsubsection`, `section_path` (`""` when empty).
5. **`CHUNK_SIZE` 1,000 → 2,000** (includes the context line), so a list stays with the sentence that explains it.
   `CHAT_NUM_CTX` 4,096 → 6,144 to fit four bigger chunks.

Later in Stage 2 (step 2.5, scaling):

6. **`s4` uses LangChain's indexing API** (`index()` + `SQLRecordManager`, `cleanup="scoped_full"`,
   `source_id_key="source"`, `key_encoder="sha256"`), called once per article. Re-runs only embed new or
   changed chunks; the logbook is `stage2/record_manager.db`. IDs are now hashes instead of `"<title>#<n>"`.
7. **`s2` removed**: it only wrote cleaned copies for reading. The pipeline is `s1` → `s4`; `s3` is optional inspection.
8. **21 more Bangladesh articles** (country article, 14 "usable", 6 well-known outlines): 47 articles in total.
9. **Streaming loader**: `load_documents()` became `WikivoyageLoader(BaseLoader)`. `s4` uses `lazy_load()`
   (one article in memory at a time) and `count()` for progress; `s3` and `explore_sections` use `load()`.

## Numbers

| | Stage 1 | Stage 2 |
|---|---|---|
| Chunks | 1,910 | **1,211** |
| Average chunk length | ~680 chars | **1,101 chars** (~275 tokens) |
| Chunks over `CHUNK_SIZE` | — | **0** |
| Tiny chunks (< 60 chars of text) | 49, all bare headings | **6, all real content** |
| Chunks not naming their place | ~52% | **0%** |
| Index build time | 35m 48s | **39m 13s** (similar total text; longer chunks cost a bit more per character) |

Chunk-size check before building (same pipeline, `CHUNK_SIZE` 1,000 vs 2,000): 2,092 vs 1,211 chunks.
Only 2,000 kept the visa-free sentence ("may enter Thailand for tourism visa-free...") in the same chunk as the country list.

### Step 2.5: indexing API runs

| Run | Articles | Chunks | Added (embedded) | Skipped | Deleted | Time |
|---|---|---|---|---|---|---|
| Rebuild (`REBUILD = True`) | 26 | 1,211 | 1,211 | 0 | 0 | 37m 06s |
| + 21 Bangladesh articles | 47 | 1,623 | **412** | **1,211** | 0 | 8m 14s |
| Same data again (streaming loader) | 47 | 1,623 | **0** | **1,623** | 0 | **3s** |
| All of Thailand + Bangladesh (rule-based `s1`) | 350 | 5,909 | **4,286** | **1,623** | 0 | 95m 28s |

Every run ended with "now in Chroma" = chunks produced (no duplicates, nothing lost).
The old 26 articles took ~0.1 s each to skip; only new chunks reached the embedding model.
The 350-article run averaged ~1.3 s per new chunk. It produced more chunks than estimated from text size
(5,909 vs ~4,700): the new articles are mostly short outlines with many small sections, so fewer characters per chunk.

### Step 2.5: test run on 47 articles (`run_2026-11-20_1521`, 21 questions, 17.8 min)

- **No regressions from adding 21 articles:** 13 of the first 14 questions got identical sources and answers.
- **7 new Bangladesh questions:** retrieval 4 ✅ #1 · 3 ⚠️ #2; answers 5 ✅ · 2 ⚠️ · 0 ❌.
- **All 21:** retrieval 11 ✅ · 8 ⚠️ · 1 ❌ · 1 by hand; answers 15 ✅ · 6 ⚠️ · 0 ❌.

### Step 2.5: final run on 350 articles (`run_2026-11-20_1723`, 21 questions, 17.1 min)

Two changes at once: 350 instead of 47 articles, and the `@place` fix. Their effects are easy to tell apart.

- **`@place` fix (3 questions): better scores, mixed ranking.** Distances dropped from ~1.3 to ~0.7
  (`dhaka_vague` 1.287 → 0.768, `coxs_bazar_sleep` 1.312 → 0.687, `bangkok_street_food` 0.894 → 0.671), so these
  no longer look like "nothing found". Dhaka now gets `Understand` #1, Bangkok `Eat > Street food` #1. But Cox's
  Bazar lost its `Sleep` chunks: inside a filtered article every chunk starts with the place name, so adding it
  makes all chunks look alike and drowns out "where can I stay".
- **Bigger index (other 18): more competitors with the same section names.** Generic, country-level questions now
  match small articles' identically named sections: three national parks' `Climate` beat `Thailand > Climate`;
  `Ko Yao > Get in` (boats from Phuket) beats `Phuket > Get in > By boat`; `Ban Rak Thai` (a tea village in Thailand)
  beats Srimangal for "tea gardens in **Bangladesh**"; `Laem Yai > Eat` beats Yaowarat for "street food in **Yaowarat**".
  The embedding matches the *topic* strongly and the *place/country* only weakly.
- **Answers held up anyway:** 16 ✅ · 5 ⚠️ · 0 ❌ (was 15 · 6 · 0), because the right chunk was usually still in the top 4.
  Retrieval: 10 ✅ · 7 ⚠️ · 3 ❌ (was 11 · 8 · 1).

**Findings in short** (one example each, with the proposed fix):

| # | Finding | Example | Proposed fix |
|---|---|---|---|
| 1 | Questions naming something specific didn't change | "Sajek Valley from Dhaka": identical sources and answer | none needed |
| 2 | **Tiny chunks win** | "street food in Yaowarat" → #1 `Laem Yai > Eat`: "There is a minimart nearby for snacks." (64 chars) | merge sections under ~200 chars into a neighbour when chunking |
| 3 | **Topic beats place** | "tea gardens in Bangladesh" → a tea village in Thailand (Ban Rak Thai) above Srimangal | `country` metadata + filter; hybrid search (BM25) |
| 4 | `@place` fix: better scores, one worse ranking | `@cox's bazar: where can I stay?`: 1.31 → 0.69, but `Sleep` left the top 4 | reranker |
| 5 | The model reads better than search finds | "best time to visit Thailand": right chunk only #4, answer still correct | fix retrieval first, not the model |
| 6 | The automatic check is too strict | `bangkok_street_food` ❌, yet `Eat > Street food` #1 and the answer is ✅ | several acceptable sources per question |

## Retrieval: Stage 1 vs Stage 2 (`s5_search`)

| Question | Stage 1 | Stage 2 |
|---|---|---|
| where can I eat biryani in Old Dhaka? | Good: top 2 biryani listings (0.568) | **Better:** top = `Eat > Dhakaiya Biryani` (0.480); all 4 relevant |
| do I need a visa for Thailand? | ❌ visa-free list not retrieved | ✅ **#1 = visa-free list** (0.722), #2 = visa-on-arrival rules (0.772) |
| @bangkok: street food at night | ❌ 3 of 4 = bare headings | ⚠️ all 4 real food content, but Yaowarat (Stage 1's #1) is missing; scores weaker (0.89–1.02) |
| cheap hotel near Khao San Road | Good | Good: all 4 from Khao San Road, 3 from `Sleep` (0.525–0.642) |
| @dhaka: how is it? | Weak (~1.0) | Weak (1.29–1.40), top hit is the tiny `Drink: Nope. Not in Dhaka.` |
| medical clinics in Chiang Mai | not tested | ✅ `Cope > Medical` (0.485), `Cope > Dental` #3 |
| how do I get from Phuket to Ko Phi Phi? | Good: Phuket ferry chunk #1 (0.591) | ⚠️ ferry chunk `Get in > By boat` is **#2** (0.601), behind `Get in > By car` (0.533) |

## Answers: Stage 1 vs Stage 2

From the automatic run `python -m stage2.s7_run_questions` (14 questions from `questions.yaml`, 9.8 minutes,
12–68 s per answer). Full report: [`results/run_2026-11-16_2141.md`](results/run_2026-11-16_2141.md).
Every answer was checked by hand against the exact chunks the model received.

| Question | Stage 1 | Stage 2 |
|---|---|---|
| biryani in Old Dhaka | ❌ merged two places (*Haji Biryani* with *Hajjee Beryani*'s address) | ✅ **fixed:** listed separately, correct details; flags Star Hotel as Dhanmondi, not Old Dhaka |
| visa for Thailand | ❌ read the visa-on-arrival list as visa-free | ✅ **fixed:** visa-free (60 days, e.g. Australia, UK, US) vs visa-on-arrival (2,000 baht, 30 days) kept apart |
| @bangkok: street food at night | ⚠️ correct but thin | ⚠️ grounded (Train Night Market Ratchada, Ramkhamhaeng Night Market, Khao San) but no Yaowarat |
| cheap hotel near Khao San Road | — | ⚠️ gave *Bella Bella House* the "no drinking water or soap" note from *Bella Bella Riverview*; called mid-range Khaosan Palace "budget" |
| @dhaka: how is it? | — | ✅ "I couldn't find that in the travel guides" |
| medical clinics in Chiang Mai | — | ✅ two clinics, correct addresses and hours |
| how do I get from Phuket to Ko Phi Phi? | ✅ ferry, 90–120 min, prices | ✅ ferry from Rassada Port, 90–120 min |
| ferry from Phuket to Ko Phi Phi | — | ⚠️ times / duration / pier fee correct; prices muddled ("300 to 600") |
| beaches near Krabi | — | ✅ Had Yao, Ao Nang, Rai Leh (thin) |
| street food in Yaowarat | — | ✅ all details correct |
| can I vape in Thailand? | — | ⚠️ right bottom line; one false claim ("no vape shops"; source says sold openly) and two citations swapped |
| best time to visit Thailand | — | ✅ Nov–Feb cool season |
| how do I greet people in Thailand? | — | ✅ the wai and its rules, correctly cited |
| capital of France | ✅ refused | ✅ refused |

**Totals (Stage 2):** retrieval 7 ✅ / 5 ⚠️ / 1 ❌ / 1 judged by hand · answers 10 ✅ / 4 ⚠️ / 0 ❌.

## The good

- **Both Stage 1 chunking problems are fixed:** no bare-heading chunks, and every chunk names its place and section.
- **The visa-free list is now retrieved first** for the visa question, together with the visa-on-arrival rules.
- **Better scores where content is specific:** biryani 0.568 → 0.480; Chiang Mai clinics 0.485.
- **Section filtering is now possible:** every chunk has a `section` key (e.g. only `Eat` sections).
- **Only `chunking.py` had to change**: extracting, cleaning, indexing and answering code was untouched.
- **Both Stage 1 answer failures are fixed:** biryani places are no longer merged, and the visa answer separates visa-free from visa-on-arrival.
- **No wrong answers out of 14**; the model refuses correctly when retrieval is weak (`@dhaka: how is it?`, capital of France).
- **Automatic test runner** (`s7_run_questions.py` + `questions.yaml`): one command, a dated report, and an automatic check of whether the expected source was retrieved.

## The bad

- **Bigger chunks can blur specific topics:** Yaowarat street food (Stage 1's best Bangkok result) dropped out of the top 4.
- **Context lines make chunks in the same section look alike:** for "how do I get from Phuket to Ko Phi Phi?", every `Get in` chunk matches, and `By car` outranks `By boat`.
- **Tiny chunks still attract vague questions:** `Drink: Nope. Not in Dhaka.` is the top hit for "@dhaka: how is it?" (still correctly flagged as weak by distance).
- **Near-duplicate text across articles:** the same Dhakaiya Biryani paragraph is in both `Dhaka` and `Dhaka/Old Dhaka`, using two result slots.
- **Indexing speed unchanged** (~39 min for 26 articles). Power-setting changes made no visible difference.
- **The 4B model still makes generation errors** with good excerpts: merged two neighbouring hotel listings (Bella Bella House / Riverview), muddled ferry prices, contradicted its source once (vaping), swapped citations.
- **Section intros and neighbouring sections can outrank the exact section:** `Khao San Road` intro beat `Sleep`; `Respect > What to do` beat `The wai`; `Yaowarat > Eat` only #4 for "street food in Yaowarat".

## Still open

- **Topic beats place in a big index:** fix with a country filter (needs a `country` metadata field: adding it
  re-embeds every chunk, ~2 h), hybrid search (BM25 matches "Yaowarat" / "Bangladesh" exactly) and a reranker → **Stage 4**
- **`@place` within one article:** keep the place in the search text (good distances) but rank within the article
  by the question alone, or let a reranker decide (Cox's Bazar `Sleep`) → **Stage 4**

- Generation errors (merged listings, numbers, citations): try a 7–8B chat model and a stricter prompt → **Stage 3/4**
- Re-rank results so the most relevant chunk within a section wins (By boat vs By car) → **Stage 4** (reranker)
- Possibly merge tiny sections into a neighbour; de-duplicate near-identical chunks
- Cleaning glitch: Thailand `Work` section starts "Thailand. Thailand has also..." (a dropped template)
- ~~Incremental/resumable indexing~~ and ~~streaming~~: done (step 2.5)
- Choose which articles to index for the full run (all ~35k at ~1.2–1.8 s/chunk is days) → **step 2.5c**
- **`@place:` drops the place name from the search text:** `@cox's bazar: where can I stay?` searches for just "where can I stay?", so even the right `Sleep` chunks score weak (1.31–1.35). Keeping the place in the query (e.g. "Cox's Bazar: where can I stay?") should help
- **Junk in the source data:** a spam listing ("All Result BD — free, 9am–5pm") in Cox's Bazar `Sleep` was passed on as a budget hotel; listings with no address/phone/description could be dropped when cleaning
- **Underscore in a context line:** Bangladesh's parent is written `South_Asia`, so its chunks start "Bangladesh (South_Asia) > ...". Fixing it in `build_metadata` re-embeds those chunks (the text changes)
- Numbers still get twisted by the 4B model (Sundarbans: a wildlife centre's fee presented as park entry; "no less than" → "as much as")
- An article deleted from Wikivoyage is never cleaned up by `scoped_full` (needs an occasional `cleanup="full"` run)
