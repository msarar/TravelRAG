# Test questions

The same questions, re-run after every stage, so we can see what got better or worse.

- **The questions themselves** live in [`questions.yaml`](questions.yaml) (with where each answer *should* come from).
- **Run them all automatically** with `python -m stage2.s7_run_questions` → full report in `stage2/results/`.
- **This file** is the human summary across stages. Details per stage: `stage1/RESULTS.md`, `stage2/RESULTS.md`.

Distances are Chroma's squared L2: **smaller is better** (~0.5 strong, ~1.0 weak).
✅ good · ⚠️ partly · ❌ wrong · — not tested

Latest Stage 2 run: [`stage2/results/run_2026-11-16_2141.md`](stage2/results/run_2026-11-16_2141.md)
(Stage 1 results were collected by hand in `s5_search` / `s6_ask`; Stage 1 had no automatic runner.)

## Retrieval: did the right chunks come back?

| id | Question | Expected source | Stage 1 | Stage 2 |
|---|---|---|---|---|
| biryani | where can I eat biryani in Old Dhaka? | Dhaka/Old Dhaka › Eat | ✅ biryani listings #1 (0.568) | ✅ `Eat > Dhakaiya Biryani` #1 (0.480) |
| visa | do I need a visa for Thailand? | Thailand › Entry requirements | ❌ visa-free list not retrieved | ✅ visa-free list #1 (0.722), visa-on-arrival #2 |
| bangkok_street_food | @bangkok: street food at night | Yaowarat › Eat | ❌ 3 of 4 bare headings | ❌ 4 real food chunks, but Yaowarat not in top 4 (0.89–1.02) |
| khao_san_hotel | cheap hotel near Khao San Road | Khao San Road › Sleep | ✅ budget chunks #1 (0.568) | ⚠️ article intro #1 (0.525), `Sleep` #2–4 |
| dhaka_vague | @dhaka: how is it? | Dhaka (any) | ⚠️ weak (~1.0), bare headings | ⚠️ weak (1.29), tiny `Old Dhaka > Drink` #1 |
| chiang_mai_clinics | medical clinics in Chiang Mai | Chiang Mai › Medical | — | ✅ `Cope > Medical` #1 (0.485) |
| phuket_phi_phi_how | how do I get from Phuket to Ko Phi Phi? | › By boat | ✅ ferry chunk #1 (0.591) | ⚠️ `By boat` #2 (0.601), behind `By car` (0.533) |
| phuket_phi_phi_ferry | ferry from Phuket to Ko Phi Phi | › By boat | — | ✅ `By boat` #1 (0.582) |
| krabi_beaches | beaches near Krabi | (judge by hand) | — | ⚠️ Krabi Town intro #1 (0.727); no Ao Nang chunks |
| yaowarat_food | street food in Yaowarat | Yaowarat › Eat | — | ⚠️ all 4 from Yaowarat, `Eat` #4 (0.741) |
| vaping | can I vape in Thailand? | Thailand › Vaping | — | ✅ `Stay safe > Vaping` #1 (0.475) |
| best_time | best time of year to visit Thailand? | Thailand › Climate | — | ✅ `Understand > Climate` #1 (0.644) |
| greeting | how do I greet people in Thailand? | Thailand › The wai | — | ⚠️ `Respect > The wai` #2 (0.710), behind `What to do` |
| off_topic | what is the capital of France? | (not covered) | ✅ correctly weak (0.971) | ✅ correctly weak (1.283) |

**Stage 2 retrieval:** 7 ✅ expected source #1 · 5 ⚠️ #2–4 · 1 ❌ · 1 judged by hand.

## Answers: correct and faithful to the excerpts?

Checked by hand against the exact chunks the model received.

| id | Stage 1 | Stage 2 |
|---|---|---|
| biryani | ❌ merged two places: *Haji Biryani* given *Hajjee Beryani*'s address | ✅ **fixed:** Haji Biryani and Hajjee Beryani listed separately, each with correct details; flags that Star Hotel is in Dhanmondi, not Old Dhaka |
| visa | ❌ read the visa-on-arrival list as visa-free | ✅ **fixed:** visa-free (e.g. Australia, UK, US, 60 days) vs visa-on-arrival (2,000 baht, 30 days) kept apart |
| bangkok_street_food | ⚠️ correct but thin | ⚠️ grounded (Train Night Market Ratchada, Ramkhamhaeng Night Market, Khao San), but misses Yaowarat because retrieval did |
| khao_san_hotel | — | ⚠️ **merged two listings:** gave *Bella Bella House* the "no drinking water or soap" note of *Bella Bella Riverview*; called mid-range Khaosan Palace "budget" |
| dhaka_vague | — | ✅ "I couldn't find that in the travel guides" |
| chiang_mai_clinics | — | ✅ CM Mediclinic and Loi Kroh Clinic with correct addresses and hours |
| phuket_phi_phi_how | ✅ ferry, 90–120 min, 600/1,000 baht | ✅ ferry from Rassada Port, 90–120 min (no prices this time) |
| phuket_phi_phi_ferry | — | ⚠️ times, duration, 20-baht fee correct; **prices muddled** ("300 to 600"; source: 600 one-way / 1,000 return at the pier, 300–350 as a guesthouse deal) |
| krabi_beaches | — | ✅ Had Yao, Ao Nang, Rai Leh (correct, but thin) |
| yaowarat_food | — | ✅ every detail correct (three generations, sunset queues, ~50 baht) |
| vaping | — | ⚠️ bottom line right (illegal), but **one false claim** ("no vape shops"; source: *sold openly in Bangkok and Pattaya*) and **two citations swapped** |
| best_time | — | ✅ Nov–Feb cool season, busy around Christmas / Chinese New Year |
| greeting | — | ✅ the wai and its rules, all correct and correctly cited |
| off_topic | ✅ "I couldn't find that in the travel guides" | ✅ same |

**Stage 2 answers:** 10 ✅ · 4 ⚠️ · 0 ❌. The remaining errors are all **generation** errors from the 4B model
(merging neighbouring listings, smoothing over numbers, swapped citations), not missing information.
