# Test questions

The same questions, re-run after every stage, so we can see what got better or worse.

- **The questions themselves** live in [`questions.yaml`](questions.yaml) (with where each answer *should* come from).
- **Run them all automatically** with `python -m stage2.s7_run_questions` → full report in `stage2/results/`.
- **This file** is the human summary across stages. Details per stage: `stage1/RESULTS.md`, `stage2/RESULTS.md`.

Distances are Chroma's squared L2: **smaller is better** (~0.5 strong, ~1.0 weak).
✅ good · ⚠️ partly · ❌ wrong · — not tested

Latest Stage 2 run: [`stage2/results/run_2026-11-20_1723.md`](stage2/results/run_2026-11-20_1723.md) (**350 articles**, `@place` fix): see [the last section](#stage-2-final-350-articles--place-fix).
The two tables below are the 47-article run [`run_2026-11-20_1521.md`](stage2/results/run_2026-11-20_1521.md).
First run: [`run_2026-11-16_2141.md`](stage2/results/run_2026-11-16_2141.md) (26 articles, first 14 questions).
Going from 26 to 47 articles changed nothing for the first 14 questions: 13 got identical sources and answers;
`biryani` gained `Bangladesh > Eat` at #4 and gave a shorter (still correct) list.
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
| saint_martin_ferry | Teknaf → Saint Martins Island? | Saint Martins Island › Get in | — | ✅ `Get in > By bus` #1 (0.560), Teknaf `Get in` #2 |
| bd_visa_on_arrival | visa on arrival in Bangladesh, cost? | Bangladesh › Visa On Arrival | — | ✅ `Visas > Visa On Arrival` #1 (0.363), all 4 from `Visas` |
| sundarbans_from_khulna | visit the Sundarbans from Khulna? | Sundarbans › Get in | — | ⚠️ `Get in` #2 (0.524), behind `Khulna > Go next` (0.515) |
| sundarbans_permit | permit for the Sundarbans? | Sundarbans (any) | — | ✅ all 4 from Sundarbans; `Fees and permits` #3 |
| tea_gardens | tea gardens in Bangladesh? | Srimangal (any) | — | ⚠️ Srimangal #2 (0.804); #1 `Sylhet > See > Further afield` (0.563) is about Sri Mangal too |
| sajek_from_dhaka | Sajek Valley from Dhaka? | Sajek Valley › From Dhaka | — | ✅ `Get in > From Dhaka` #1 (0.439) |
| coxs_bazar_sleep | @cox's bazar: where can I stay? | Cox's Bazar › Sleep | — | ⚠️ `Sleep` #2 and #4, but **all weak (1.31–1.35)** |

**Stage 2 retrieval (21 questions):** 11 ✅ expected source #1 · 8 ⚠️ #2–4 · 1 ❌ · 1 judged by hand.

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
| saint_martin_ferry | — | ✅ ships at 9–9:30AM, 1½–2 h, names, fares, return ~3PM, plus Keari Sindbad's own port (from Teknaf `Get in`) |
| bd_visa_on_arrival | — | ✅ US$51 / €51 in cash, US$500, onward ticket, hotel booking, not by train, 30 days + 30 |
| sundarbans_from_khulna | — | ⚠️ tours, prices, permit and Mongla boats right, but **two numbers twisted**: Tk 750/day is the *Karamjal center's* fee, not park entry; boatmen want "no less than" Tk 200, not "as much as" |
| sundarbans_permit | — | ✅ yes, from the Divisional Forest Office, Circuit House Road, Khulna |
| tea_gardens | — | ✅ Sri Mangal, "land of two leaves and a bud", Tea Research Institute (treats *Sri Mangal* and *Sreemangal* as two nearby places: they're spellings of one) |
| sajek_from_dhaka | — | ✅ bus to Khagrachhari, then jeep/CNG/motorbike, 2–3 h (no convoy mention: that chunk was #4) |
| coxs_bazar_sleep | — | ⚠️ real hotels with prices, but lists **"All Result BD — free, 9am–5pm"** (a junk listing in the guide itself) as a budget option |

**Stage 2 answers (21 questions):** 15 ✅ · 6 ⚠️ · 0 ❌ (first 14: 10 ✅ · 4 ⚠️, unchanged). The remaining errors are all **generation** errors from the 4B model
(merging neighbouring listings, smoothing over numbers, swapped citations), not missing information.

## Stage 2 final: 350 articles + `@place` fix

Run [`run_2026-11-20_1723.md`](stage2/results/run_2026-11-20_1723.md): all of Thailand + Bangladesh (350 articles, 5,909 chunks),
and `@place:` questions now search with the place name kept in (`"Cox's Bazar: where can I stay?"`).
10 of 21 questions got identical sources and answers. The ones that changed:

| id | 47 articles | 350 articles + fix | Why |
|---|---|---|---|
| bangkok_street_food | ❌ Yaowarat missing, weak (0.89–1.02) / answer ⚠️ | ❌ Yaowarat still missing, but `Bangkok > Eat > Street food` #1 (**0.671**) / answer ✅ | `@place` fix: much closer matches |
| dhaka_vague | ⚠️ tiny `Drink` chunk #1 (1.287) / ✅ refused | ✅ `Dhaka > Understand` #1 (**0.768**) / ✅ a real overview | `@place` fix |
| coxs_bazar_sleep | ⚠️ `Sleep` #2 (1.337) / ⚠️ | ❌ no `Sleep` chunk; `See`, `Understand`, `Drink` (~0.69) / ⚠️ only "Hotel Motel Zone" + hotel names from the bar list | `@place` fix: the place name now dominates, and every chunk of the article starts with it |
| phuket_phi_phi_how | ⚠️ `By boat` #2 / ✅ | ⚠️ `By boat` #4, behind `Ko Yao` and `Ko Bon` `Get in` / ✅ (thin) | new articles that also mention boats from Phuket |
| phuket_phi_phi_ferry | ✅ #1 / ⚠️ prices muddled | ⚠️ #2 behind `Ko Yao > Get in` / ✅ no prices this time | new articles |
| krabi_beaches | 👀 / ✅ | 👀 `Krabi Province > Understand` now in / ✅ richer | new articles (helped) |
| yaowarat_food | ⚠️ `Eat` #4 / ✅ | ❌ `Laem Yai > Eat` #1, `Central Thailand > Eat` #4 / ⚠️ thinner, no Eat details | new articles: other `Eat` sections |
| best_time | ✅ #1 / ✅ | ⚠️ #4 behind three national parks' `Climate` / ✅ (adds the Gulf Coast exception, correctly) | new articles: many `Climate` sections |
| tea_gardens | ⚠️ #2 / ✅ | ⚠️ Srimangal #4 behind `Ban Rak Thai` (a tea village in **Thailand**) / ✅ | new articles: topic beats country |
| visa, off_topic | ✅ / ✅ | ✅ / ✅ (one weaker slot each replaced by a new article) | |

**Retrieval (21):** 10 ✅ · 7 ⚠️ · 3 ❌ · 1 by hand (47 articles: 11 · 8 · 1 · 1).
**Answers (21):** **16 ✅ · 5 ⚠️ · 0 ❌** (47 articles: 15 · 6 · 0): answers held up because the right chunk was usually still somewhere in the top 4.
