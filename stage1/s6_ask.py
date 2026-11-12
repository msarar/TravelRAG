"""Stage 1, piece 6: the full RAG loop. Retrieve chunks, then let the LLM answer from them.

    question
       │
       ▼
    RETRIEVE   find the TOP_K closest chunks in Chroma          (step 5)
       │
       ▼
    AUGMENT    paste those chunks into a prompt with the question
       │
       ▼
    GENERATE   the chat model writes an answer from that prompt

The key idea: the LLM is told to answer ONLY from the pasted excerpts, not from
its own memory. Remember when qwen3 put Shah Jalal Dargah in Dhaka? Grounding
the answer in retrieved text is how RAG prevents that.

This file contains the same pipeline written TWO ways:
  - answer_step_by_step(): plain Python, one step at a time (read this first)
  - answer_with_lcel():    LangChain's "LCEL" style, where steps are joined with |
Switch between them with USE_LCEL below. They should give the same answers.

HOW TO USE (type at the prompt):
    where can I eat biryani in Old Dhaka?
    @bangkok: street food at night        -> only search chunks from Bangkok
    quit

Run (from the project root):  python -m stage1.s6_ask
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from common.query_parsing import list_places, parse_input, resolve_place
from common.vector_store import open_vector_store
from stage1.config import (
    CHAT_MODEL,
    CHAT_NUM_CTX,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_NUM_CTX,
)

# How many chunks to give the LLM for each question.
TOP_K = 4

# If even the BEST chunk is further away than this, retrieval probably found
# nothing relevant (see the "@Dhaka: how is it?" test: all ~1.0).
WEAK_DISTANCE = 0.95

# False = use answer_step_by_step(),  True = use answer_with_lcel()
USE_LCEL = True


# ---------------------------------------------------------------------------
# The prompt: instructions for the model + a slot for the excerpts and question
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are a travel assistant. Answer the question using ONLY the travel guide excerpts provided.

Rules:
- If the excerpts do not contain the answer, say: "I couldn't find that in the travel guides." Do not use outside knowledge and do not make things up.
- Be concise and practical. Include names, addresses and prices when the excerpts give them.
- After each fact, cite the excerpt it came from by its number, like [1] or [2]."""

QUESTION_TEMPLATE = """Travel guide excerpts:

{context}

Question: {question}"""

# A ChatPromptTemplate is a reusable prompt with {placeholders}.
# Filling it in (prompt.invoke) produces the list of messages sent to the model:
#   - a "system" message: instructions the model should follow
#   - a "human" message:  what the user is asking (here: excerpts + question)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTIONS),
    ("human", QUESTION_TEMPLATE),
])


def format_context(docs) -> str:
    """Turn retrieved Documents into one numbered block of text for the prompt.

    [1] Dhaka/Old Dhaka
    - Hajjee Beryani — 70 Kazi Alauddin Road ...

    [2] Dhaka
    ...
    The numbers let the model cite its sources, and let us check them.
    """
    blocks = []

    for number, doc in enumerate(docs, start=1):
        title = doc.metadata["title"]
        block = f"[{number}] {title}\n{doc.page_content}"
        blocks.append(block)

    # Blank lines between excerpts so the model can tell where each one ends.
    return "\n\n".join(blocks)


def print_sources(results):
    """List what the answer was based on, with distances, so you can check it."""
    print("\n\nSources:")
    for number, (doc, distance) in enumerate(results, start=1):
        title = doc.metadata["title"]
        url = doc.metadata["source"]
        print(f"  [{number}] {title}  (distance {distance:.3f})  {url}")


# ---------------------------------------------------------------------------
# Version 1: step by step, in plain Python
# ---------------------------------------------------------------------------

def answer_step_by_step(vector_store, llm, question, search_filter):
    # 1. RETRIEVE: the same call as in s5_search.py.
    results = vector_store.similarity_search_with_score(
        question,
        k=TOP_K,
        filter=search_filter,
    )

    if not results:
        print("No chunks found.")
        return

    # Warn if even the best match is weak: the answer will probably be poor.
    best_distance = results[0][1]
    if best_distance > WEAK_DISTANCE:
        print(f"(Note: best match is weak, distance {best_distance:.3f}. "
              f"The guides may not cover this.)\n")

    # 2. AUGMENT: build the prompt.
    docs = []
    for doc, _distance in results:
        docs.append(doc)

    context = format_context(docs)

    # Fill in the template's {context} and {question} placeholders.
    messages = prompt.invoke({"context": context, "question": question})

    # 3. GENERATE: stream the answer, printing each piece as it arrives.
    #    (invoke() would wait for the whole answer; at ~8 tokens/second,
    #     streaming lets you start reading straight away.)
    for piece in llm.stream(messages):
        print(piece.content, end="", flush=True)

    print_sources(results)


# ---------------------------------------------------------------------------
# Version 2: the same pipeline in LangChain Expression Language (LCEL)
# ---------------------------------------------------------------------------

def answer_with_lcel(vector_store, llm, question, search_filter):
    # A retriever is a standard LangChain object: give it a question, get back
    # a list of Documents. as_retriever() wraps our vector store as one.
    retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K, "filter": search_filter},
    )

    # In LCEL, "a | b" means: "run a, then feed its output into b".
    # This chain does exactly what answer_step_by_step does:
    #
    #   {                                       <- run both of these on the question:
    #     "context":  retriever | format,          question -> Documents -> one text block
    #     "question": RunnablePassthrough(),       question -> question (unchanged)
    #   }                                       -> gives {"context": ..., "question": ...}
    #   | prompt                                -> fills the template = the messages
    #   | llm                                   -> the model's reply (an AIMessage)
    #   | StrOutputParser()                     -> just the reply's text
    chain = (
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # .stream() works on the whole chain: the answer arrives piece by piece.
    for piece in chain.stream(question):
        print(piece, end="", flush=True)

    # The chain only returns text, not the retrieved chunks, so this version
    # can't print the sources. (Later we'll build chains that return both.)
    print()


# ---------------------------------------------------------------------------

def main():
    print("Opening the index...")
    vector_store = open_vector_store(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_num_ctx=EMBEDDING_NUM_CTX,
    )
    places = list_places(vector_store)

    # The chat model. temperature=0 keeps answers as factual and repeatable
    # as possible: we want it to stick to the excerpts, not be creative.
    llm = ChatOllama(model=CHAT_MODEL, temperature=0, num_ctx=CHAT_NUM_CTX)

    version = "LCEL chain" if USE_LCEL else "step by step"
    print(f"Ready ({version}). Ask a question, '@place: question' to filter, or 'quit'.\n")

    while True:
        user_input = input("question> ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break
        if not user_input.strip():
            continue

        question, typed_place = parse_input(user_input)

        search_filter = None
        if typed_place is not None:
            place = resolve_place(typed_place, places)
            if place is None:
                print(f"Unknown place {typed_place!r}. Choose from: {', '.join(places)}\n")
                continue
            search_filter = {"place": place}

        print()
        if USE_LCEL:
            answer_with_lcel(vector_store, llm, question, search_filter)
        else:
            answer_step_by_step(vector_store, llm, question, search_filter)
        print()


if __name__ == "__main__":
    main()
