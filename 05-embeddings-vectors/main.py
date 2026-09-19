# Embeddings and vector similarity
# --------------------------------
# This script takes a set of sentences, converts each into a vector (a list of 384 numbers) using a small embedding model downloaded from Hugging Face, then measures how "close" a new query is to each sentence using cosine similarity. No LLM involved at any point, this is pure meaning-lookup, the mechanism underneath RAG.

## Current problem with this script:
# --------------------------------
# The list doesn't leave in a memory, it cannot survive a second run. It re-embeds every sentence, every time, from scratch,fine for eight test sentences, completely unworkable for 25 years of service reports, which you'd want embedded once and queried forever after. That's the entire justification for a vector store: it's not a smarter search algorithm, it's persistence, the same reason you'd never keep a real application's data in a Python list instead of a database.
#
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import numpy as np

# SentenceTransformer knows how to load the model from Hugging face, it downloads once, then cached locally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# a small mixed set: some field-service-style sentences, some unrelated
sentences = [
    "No pressure on the lift arm after cold start",
    "Hydraulic pressure loss in the main cylinder",
    "Valve fails to hold charge overnight",
    "Electrical fault in the control panel wiring",
    "Motor overheats under continuous load",
    "The invoice was sent to the wrong customer",
    "Quarterly sales report is due next Friday",
    "The cat sat on the windowsill in the sun",
]

# Turn every sentence into a vector, this is the actual embedding step
embeddings = model.encode(sentences)

print(f"Each sentence became a vector of length: {embeddings[0].shape}\n")

# A new query, not in the list above
query = "lift arm won't build pressure"
query_embedding = model.encode(query)

# Cosine similarity, the standard way to measure "how close" two vectors are
# ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


scores = [cosine_similarity(query_embedding, e) for e in embeddings]
# scores =[]
# for i, e in enumerate(embeddings):
#   score= cosine_similarity( query_embedding, e)
#   print(f"iteration {i}: {score:.3f}  {sentences[i]}")
#   scores.append(score)

# Rank sentences by similarity, best first
ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

print(f"Query: '{query}'\n")
for sentence, score in ranked:
    print(f"{score:.3f}  {sentence}")

# Outcome of this script:
# ----------------------
# main⚡ ⇒ uv run 05-embeddings-vectors/main.py
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 27673.65it/s]
# Each sentence became a vector of length: (384,)
# Query: 'lift arm won't build pressure'

# 0.768  No pressure on the lift arm after cold start
# 0.314  Hydraulic pressure loss in the main cylinder
# 0.228  Valve fails to hold charge overnight
# 0.194  Motor overheats under continuous load
# 0.077  The invoice was sent to the wrong customer
# 0.027  Quarterly sales report is due next Friday
# 0.024  Electrical fault in the control panel wiring
# -0.001  The cat sat on the windowsill in the sun