from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    GROQ_API_KEY,
)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Groq LLM client
client = Groq(api_key=GROQ_API_KEY)

PROMPT = """
You MUST answer using ONLY the provided sources.
If the answer is not found in the sources, reply: "I don't know".

Sources:
{contexts}

Question: {question}
"""

def retrieve(query):
    q_emb = embed_model.encode(query).tolist()
    results = index.query(vector=q_emb, top_k=5, include_metadata=True)
    return results.get("matches", [])

def build_context(matches):
    blocks = []
    for m in matches:
        text = m.metadata.get("text", "")
        blocks.append(f"[{m.id}] {text}")
    return "\n\n".join(blocks)

def generate_answer(question, contexts):
    message = PROMPT.format(question=question, contexts=contexts)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": message}],
        temperature=0
    )

    return response.choices[0].message.content

def rag_answer(query):
    matches = retrieve(query)
    context = build_context(matches)
    answer = generate_answer(query, context)

    confidence = (
        sum([m.score for m in matches]) / len(matches)
        if matches else 0
    )

    # Convert Pinecone objects into serializable dictionaries
    clean_chunks = []
    for m in matches:
        clean_chunks.append({
            "id": m.id,
            "score": m.score,
            "metadata": m.metadata
        })

    return {
        "answer": answer,
        "chunks": clean_chunks,
        "confidence": confidence
    }
