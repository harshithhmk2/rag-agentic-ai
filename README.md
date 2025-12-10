# 🚀 Agentic AI RAG Chatbot
A Retrieval-Augmented Generation (RAG) chatbot built using **Python**, **LangGraph**, **Pinecone**, and **Text Embeddings**, designed to answer questions **strictly from the PDF knowledge base**:  
**Ebook-Agentic-AI.pdf**

---

## 📌 Features
- PDF ingestion → chunking → embeddings → Pinecone vector search  
- LangGraph-powered RAG pipeline (Retriever + LLM + Router)  
- API endpoints (FastAPI) OR optional Streamlit UI  
- Deterministic answers strictly grounded in the PDF  
- Clean and modular architecture

---

## 🗂️ Project Structure
```
rag-agentic-ai/
│── data/
│    └── Ebook-Agentic-AI.pdf
│── app/
│    ├── ingest.py
│    ├── rag_graph.py
│    ├── server.py
│    └── config.py
│── requirements.txt
│── README.md
```

---

# ⚙️ Setup Instructions

## 1️⃣ Clone the Repo
```bash
git clone https://github.com/<your-username>/rag-agentic-ai.git
cd rag-agentic-ai
```

---

## 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 3️⃣ Add API Keys  
Create `.env` file in root:

```
PINECONE_API_KEY=xxxx
OPENAI_API_KEY=xxxx
INDEX_NAME=agentic-ai-index
```

---

## 4️⃣ Run PDF Ingestion  
This loads the PDF → chunks text → embeds → stores in Pinecone.

```bash
python app/ingest.py
```

---

## 5️⃣ Start the API Server  
```bash
uvicorn app.server:app --reload
```

---

## 6️⃣ Query the API  
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is agentic AI?"}'
```

---

# 💬 Sample Streamlit UI (optional)

Run:
```bash
streamlit run app/ui.py
```

---

# 🧠 Example Queries (5–6 Sample Questions)

| Query | What It Tests |
|-------|---------------|
| **What is Agentic AI according to the ebook?** | Basic definition retrieval |
| **What are the key properties of agentic systems?** | Multi-chunk retrieval |
| **How do agentic workflows differ from traditional automation?** | Concept comparison |
| **What role does autonomy play in agentic architectures?** | Deep semantic understanding |
| **Explain agentic orchestration from the PDF.** | Long-form answer grounding |
| **What benefits do agentic AI systems provide to enterprises?** | Section-specific reasoning |

---

# 🏗️ Short Architecture Explanation

## 🔹 **1. Ingestion Layer**
- Load PDF using `PyPDFLoader`
- Split into 500–800 character chunks (LangChain splitter)
- Generate embeddings using **OpenAI text-embedding-3-large**
- Store vectors in **Pinecone**

## 🔹 **2. Retrieval Layer**
When a user asks a question:
- Convert query → embedding  
- Pinecone similarity search  
- Return top-k relevant chunks  

## 🔹 **3. LangGraph RAG Pipeline**
Graph Nodes:
- **Retrieval Node** – Fetch context from Pinecone  
- **LLM Node** – Generate grounded response  
- **Validation Node** (optional) – Ensure response adheres to PDF sources  

Graph guarantees:
- Deterministic flow  
- Reusable nodes  
- Easy to extend (tools, agents, memory)  

## 🔹 **4. API Layer (FastAPI)**
Endpoints:
- `/query` – Main RAG query  
- `/health` – Check server status  

## 🔹 **5. Optional UI**
Streamlit frontend → sends requests to FastAPI.

