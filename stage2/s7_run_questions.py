"""Stage 2, piece 7: answer every test question automatically and save the results.

Instead of typing questions into s6_ask.py and copying answers around, this
script reads ALL questions from questions.yaml (project root), runs each one
through the same retrieval + prompt + LLM as s6_ask.py, and writes:

    stage2/results/run_<date>_<time>.md     readable report (open in VS Code)
    stage2/results/run_<date>_<time>.json   the same data, for scripts later (Stage 3)

For each question it also checks automatically whether retrieval found the
EXPECTED source (expect_title / expect_section in questions.yaml), so you can
see at a glance which questions improved or broke.

Run (from the project root):  python -m stage2.s7_run_questions
Takes a while: ~15-60 seconds per question on CPU (the LLM writing the answer).
"""

import json
import time
from datetime import datetime

import yaml                                   # reads questions.yaml (PyYAML comes with LangChain)
from langchain_ollama import ChatOllama

from common.query_parsing import list_places, parse_input, resolve_place
from common.vector_store import open_vector_store
from stage2.config import (
    CHAT_MODEL,
    CHAT_NUM_CTX,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_NUM_CTX,
    PROJECT_ROOT,
    STAGE_DIR,
)
# Re-use EXACTLY the same prompt and settings as s6_ask.py, so the results here
# match what you'd get by asking s6_ask.py by hand. (This import works because
# the file name starts with a letter: "s6_ask", not "6_ask".)
from stage2.s6_ask import TOP_K, WEAK_DISTANCE, format_context, prompt

QUESTIONS_FILE = PROJECT_ROOT / "questions.yaml"
RESULTS_DIR = STAGE_DIR / "results"
STAGE_NAME = "Stage 2"

# How much of each retrieved chunk to show in the report.
PREVIEW_CHARS = 200

# The answer the prompt tells the model to give when the excerpts don't help.
NOT_FOUND_PHRASE = "couldn't find"


# ---------------------------------------------------------------------------
# Reading questions
# ---------------------------------------------------------------------------

def load_questions() -> list[dict]:
    """Read questions.yaml into a list of dicts like {"id": ..., "question": ..., ...}."""
    text = QUESTIONS_FILE.read_text(encoding="utf-8")
    questions = yaml.safe_load(text)          # YAML text -> Python list of dicts
    return questions


# ---------------------------------------------------------------------------
# Checking retrieval against what we expected
# ---------------------------------------------------------------------------

def describe_source(doc) -> str:
    """"Bangkok/Silom > Eat > Budget" (or just the title for an article's intro)."""
    title = doc.metadata["title"]
    section_path = doc.metadata.get("section_path", "")
    if section_path:
        return f"{title} > {section_path}"
    return title


def find_expected_rank(results, expect_title: str, expect_section: str):
    """Position (1, 2, ...) of the first retrieved chunk that matches what we expected.

    expect_title must match the chunk's title exactly (if given).
    expect_section must appear somewhere in the chunk's section_path (if given),
    so "Eat" matches both "Eat" and "Eat > Budget".
    Returns None if no retrieved chunk matches.
    """
    for rank, (doc, _distance) in enumerate(results, start=1):
        title = doc.metadata["title"]
        section_path = doc.metadata.get("section_path", "")

        title_ok = True
        if expect_title:
            title_ok = (title == expect_title)

        section_ok = True
        if expect_section:
            section_ok = (expect_section.lower() in section_path.lower())

        if title_ok and section_ok:
            return rank

    return None


def check_retrieval(q: dict, results, best_distance: float) -> str:
    """A short verdict for the report: did retrieval do what questions.yaml expected?"""
    covered = q.get("covered", True)

    # Questions the guides don't cover: retrieval SHOULD look weak.
    if not covered:
        if best_distance > WEAK_DISTANCE:
            return "✅ correctly weak"
        return "❌ looked relevant but shouldn't"

    expect_title = q.get("expect_title", "")
    expect_section = q.get("expect_section", "")

    # Nothing specific expected: a human has to judge it.
    if not expect_title and not expect_section:
        return "👀 check by hand"

    rank = find_expected_rank(results, expect_title, expect_section)
    if rank is None:
        return f"❌ expected source not in top {TOP_K}"
    if rank == 1:
        return "✅ expected source #1"
    return f"⚠️ expected source #{rank}"


def check_answer(q: dict, answer: str) -> str:
    """Only automatic check we can do safely: off-topic questions should say 'couldn't find'."""
    covered = q.get("covered", True)
    said_not_found = NOT_FOUND_PHRASE in answer.lower()

    if not covered:
        if said_not_found:
            return "✅ said it couldn't find it"
        return "❌ answered anyway"

    if said_not_found:
        return "⚠️ said it couldn't find it"
    return "👀 check by hand"


# ---------------------------------------------------------------------------
# Running one question
# ---------------------------------------------------------------------------

def run_one_question(q: dict, vector_store, llm, places: list[str]) -> dict:
    """Retrieve + answer ONE question. Returns everything we want to record about it."""
    question, typed_place = parse_input(q["question"])

    # Place filter, exactly like s5/s6 ("@bangkok: ..." -> {"place": "Bangkok"}).
    search_filter = None
    if typed_place is not None:
        place = resolve_place(typed_place, places)
        if place is not None:
            search_filter = {"place": place}

    # 1. RETRIEVE (with distances).
    results = vector_store.similarity_search_with_score(
        question,
        k=TOP_K,
        filter=search_filter,
    )
    best_distance = results[0][1] if results else 99.0

    # 2. AUGMENT: same prompt as s6_ask.py.
    docs = []
    for doc, _distance in results:
        docs.append(doc)
    context = format_context(docs)
    messages = prompt.invoke({"context": context, "question": question})

    # 3. GENERATE: invoke (not stream), because nobody is watching it type.
    start = time.time()
    reply = llm.invoke(messages)
    seconds = time.time() - start
    answer = reply.content

    # Record the sources in a plain form we can save as JSON.
    sources = []
    for doc, distance in results:
        sources.append({
            "where": describe_source(doc),
            "distance": round(distance, 3),
            "url": doc.metadata["source"],
            "text": doc.page_content,
        })

    return {
        "id": q["id"],
        "question": q["question"],
        "note": q.get("note", ""),
        "filter": search_filter,
        "best_distance": round(best_distance, 3),
        "retrieval_check": check_retrieval(q, results, best_distance),
        "answer_check": check_answer(q, answer),
        "answer": answer,
        "answer_seconds": round(seconds, 1),
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Writing the report
# ---------------------------------------------------------------------------

def write_markdown(results: list[dict], path, started: str, total_seconds: float):
    lines = []
    lines.append(f"# {STAGE_NAME}: test question run")
    lines.append("")
    lines.append(f"- Run started: {started}")
    lines.append(f"- Models: chat `{CHAT_MODEL}`, embeddings `{EMBEDDING_MODEL}`, top {TOP_K} chunks")
    lines.append(f"- Total time: {total_seconds / 60:.1f} minutes")
    lines.append("")

    # Summary table: one row per question.
    lines.append("## Summary")
    lines.append("")
    lines.append("| id | best distance | retrieval | answer |")
    lines.append("|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['id']} | {r['best_distance']} | {r['retrieval_check']} | {r['answer_check']} |")
    lines.append("")

    # Details: question, sources, answer.
    lines.append("## Details")
    for r in results:
        lines.append("")
        lines.append(f"### {r['id']}: {r['question']}")
        lines.append("")
        if r["note"]:
            lines.append(f"*{r['note']}*")
            lines.append("")
        lines.append(f"Retrieval: {r['retrieval_check']} · Answer: {r['answer_check']} · "
                     f"answered in {r['answer_seconds']}s")
        lines.append("")
        lines.append("**Sources**")
        lines.append("")
        for number, s in enumerate(r["sources"], start=1):
            preview = s["text"][:PREVIEW_CHARS].replace("\n", " ")
            lines.append(f"{number}. `{s['where']}` (distance {s['distance']})  ")
            lines.append(f"   {preview}…")
        lines.append("")
        lines.append("**Answer**")
        lines.append("")
        lines.append(r["answer"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(results: list[dict], path):
    text = json.dumps(results, indent=2, ensure_ascii=False)   # keep Thai/Bengali readable
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------

def main():
    questions = load_questions()
    print(f"{len(questions)} questions from {QUESTIONS_FILE.name}")

    print("Opening the index...")
    vector_store = open_vector_store(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_num_ctx=EMBEDDING_NUM_CTX,
    )
    places = list_places(vector_store)
    llm = ChatOllama(model=CHAT_MODEL, temperature=0, num_ctx=CHAT_NUM_CTX)

    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_start = time.time()

    results = []
    for number, q in enumerate(questions, start=1):
        print(f"  [{number}/{len(questions)}] {q['id']:<24}", end="", flush=True)
        result = run_one_question(q, vector_store, llm, places)
        results.append(result)
        print(f"{result['answer_seconds']:>5}s   {result['retrieval_check']}")

    total_seconds = time.time() - run_start

    # File names include date and time, so runs never overwrite each other.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    md_path = RESULTS_DIR / f"run_{stamp}.md"
    json_path = RESULTS_DIR / f"run_{stamp}.json"

    write_markdown(results, md_path, started, total_seconds)
    write_json(results, json_path)

    print(f"\nDone in {total_seconds / 60:.1f} minutes.")
    print(f"Report: {md_path}")
    print(f"Data:   {json_path}")


if __name__ == "__main__":
    main()
