import re

def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()

def chunk_text(text, chunk_size=800, overlap=160):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks
