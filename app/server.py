from fastapi import FastAPI
from pydantic import BaseModel
from app.langgraph_rag import rag_answer

app = FastAPI(title="RAG Agentic AI Chatbot")

class Query(BaseModel):
    q: str

@app.post("/chat")
def chat(payload: Query):
    result = rag_answer(payload.q)
    return {
        "question": payload.q,
        "answer": result["answer"],
        "chunks": result["chunks"],
        "confidence": result["confidence"],
    }
