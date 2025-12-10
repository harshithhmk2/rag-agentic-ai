import requests
import fitz
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from app.config import (
    SOURCE_PDF_URL,
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX,
)
from app.utils import clean_text, chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")


def download_pdf(url, dst="ebook.pdf"):
    r = requests.get(url)
    with open(dst, "wb") as f:
        f.write(r.content)
    return dst


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [doc[i].get_text() for i in range(len(doc))]
    return "\n".join(pages)


def embed_chunks(chunks):
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)


def main():
    print("🔌 Connecting to Pinecone...")

    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    print("📘 Downloading PDF...")
    pdf_path = download_pdf(SOURCE_PDF_URL)

    print("📖 Extracting text...")
    text = extract_text(pdf_path)
    text = clean_text(text)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print("🧠 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("⬆️ Uploading to Pinecone...")
    for chunk, emb in tqdm(list(zip(chunks, embeddings))):
        index.upsert(
            vectors=[
                {
                    "id": str(uuid.uuid4()),
                    "values": emb.tolist(),
                    "metadata": {"text": chunk},
                }
            ]
        )

    print("🎉 DONE! Uploaded all chunks.")


if __name__ == "__main__":
    main()
