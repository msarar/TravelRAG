# Test questions

Questions to re-run after every change, so we can tell whether retrieval got
better or worse. In Stage 3 this becomes a proper evaluation set with scores.

Results below are from the first index (Stage 1: 1000-char chunks, qwen3-embedding:0.6b).
Distances are Chroma's squared L2: smaller is better (~0.57 strong, ~1.0 weak).

| # | Question | Should come from | Stage 1 result |
|---|---|---|---|
| 1 | where can I eat biryani in Old Dhaka? | Dhaka/Old Dhaka, Eat | Good: top 2 are the biryani listings (0.568, 0.701) |
| 2 | do I need a visa for Thailand? | Thailand, Get in / visas | Good: all 4 from the visa section (0.79-0.89) |
| 3 | @Bangkok: street food at night | Bangkok districts, Eat | Bad: #1 good (Yaowarat), #2-4 are heading-only chunks ("### Shopping malls", "## Drink") |
| 4 | cheap hotel near Khao San Road | Bangkok/Khao San Road, Sleep | Good: top 2 budget-accommodation chunks (0.568, 0.571) |
| 5 | @Dhaka: how is it? | Dhaka, Understand | Weak: vague question; all distances ~1.0, heading-only chunks returned |
| 6 | medical clinics in Chiang Mai | Chiang Mai, Stay healthy | Not tried yet (Loi Kroh Clinic chunk lost its "### Medical" heading) |
| 7 | how do I get from Phuket to Ko Phi Phi? | Ko Phi Phi / Phuket, Get in | Good: Phuket, Patong, Ko Phi Phi chunks (0.59-0.62) |
| 8 | what is the capital of France? | (not covered) | Correctly weak: best 0.971 |

## Answers from the LLM (s6_ask.py, qwen3:4b-instruct, Stage 1)

Retrieval and generation can fail separately, so each answer is judged on its own.

| # | Answer | Verdict |
|---|---|---|
| 1 | Lists "Haji Biryani — 70 Kazi Alauddin Road" and Nanna Biryani | **Wrong (merged two places).** Excerpt [1] has *Hajjee Beryani* at 70 Kazi Alauddin Road; excerpt [2] has a different *Haji Biryani* (phone only). The model blended them. Also left out other listings. |
| 2 | Says Bulgaria, Bhutan, Cyprus... "do not need a visa" | **Wrong (misread).** That list in excerpt [1] is the *visa-on-arrival* list (pay 2,000 baht), not visa-free. The real visa-exempt list is in the chunk just before it, which was NOT retrieved (it starts "- 14 days: Myanmar..." and never says "visa": an orphan chunk). Same answer from the LCEL version. |
| 3 | Yaowarat street food, ~50 baht per dish | **OK but thin.** Used only excerpt [1]; the other 3 were heading-only chunks, so there was nothing else to use. Did not invent anything. |
| 7 | Ferry from Rassada Port, 90-120 min, 600 / 1,000 baht | **Good.** Correct and cited [1]. |
| 8 | "I couldn't find that in the travel guides." + weak-match note | **Good.** Grounding works: it did not answer "Paris". |

## Known problems (Stage 1)

- **Heading-only chunks:** 49 of 1,910 chunks are just a heading. Their vague
  vectors match many questions and push real content out. Fix: Stage 2.
- **Orphan chunks:** about half the chunks never mention their place name, and
  some lists lose their introduction (the visa-exempt list above). Fix: Stage 2.
- **Generation errors even with good excerpts:** the 4B model merged two restaurants
  and misread a visa list. Ideas to test in Stage 3/4: a 7-8B chat model, a stricter
  prompt ("copy names exactly, never combine listings"), chunks that keep a list's intro.
