"""Stage 0 smoke test: checks that embeddings and the chat model both work through LangChain.

What it does:
  1. Embeds three sentences and compares them. The two about food should score
     as more similar to each other than to the one about visas.
  2. Asks the chat model a one-line question.

Run (from the project root):  python -m stage2.s0_smoke_test
"""

import numpy as np                                          # maths on lists of numbers
from langchain_ollama import ChatOllama, OllamaEmbeddings   # LangChain's Ollama connectors

from stage2.config import CHAT_MODEL, EMBEDDING_MODEL, EMBEDDING_NUM_CTX


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How similar two vectors are, by the angle between them.

    1.0  = pointing the same way (same meaning)
    ~0   = unrelated
    Formula: (a · b) / (length of a × length of b)
    """
    dot_product = a @ b                 # "@" is the dot product for numpy arrays
    length_a = np.linalg.norm(a)        # norm = the vector's length
    length_b = np.linalg.norm(b)
    similarity = dot_product / (length_a * length_b)
    return float(similarity)


# --- 1. Embeddings ---------------------------------------------------------

# Connect to the embedding model running in Ollama.
# num_ctx=2048 limits the context window, which saves a lot of memory
# (our chunks will be much shorter than 2048 tokens anyway).
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, num_ctx=EMBEDDING_NUM_CTX)

texts = [
    "Cheap street food stalls near the old fort",      # food
    "Budget places to eat in the historic quarter",    # food, but different words
    "Visa requirements for entering the country",      # unrelated topic
]

# embed_documents returns one vector (a list of 1024 floats) per text.
vectors = embeddings.embed_documents(texts)

# Convert each plain Python list into a numpy array so we can do maths on it.
food_1 = np.array(vectors[0])
food_2 = np.array(vectors[1])
visa = np.array(vectors[2])

print("Embedding dimensions:", len(food_1))

food_vs_food = cosine_similarity(food_1, food_2)
food_vs_visa = cosine_similarity(food_1, visa)

print("food vs food :", round(food_vs_food, 3))
print("food vs visa :", round(food_vs_visa, 3))


# --- 2. Chat model ---------------------------------------------------------

# Connect to the chat model running in Ollama.
# temperature=0 makes answers as deterministic (repeatable) as possible.
llm = ChatOllama(model=CHAT_MODEL, temperature=0)

question = "In one sentence, what is retrieval-augmented generation (RAG)?"

# invoke() sends the question and waits for the full answer.
# It returns an AIMessage object; the text of the answer is in .content
reply = llm.invoke(question)

print("\nLLM says:", reply.content)
print("length of food_1 vector:", np.linalg.norm(food_1))
