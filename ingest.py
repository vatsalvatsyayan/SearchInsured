import json
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

PDF_PATH = "data/sampleinsurance.pdf"
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.json"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with page numbers."""
    print(f"📄 Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append({"page_num": i + 1, "text": text})
    print(f"✅ Extracted {len(pages)} pages")
    return pages

def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk text with overlap, preserving page numbers."""
    print(f"✂️  Chunking text (size={chunk_size}, overlap={overlap})")
    chunks = []
    
    for page in pages:
        text = page["text"]
        page_num = page["page_num"]
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Respect word boundaries
            if end < len(text) and text[end] not in [' ', '\n', '.', ',', '!', '?']:
                last_space = chunk.rfind(' ')
                if last_space > chunk_size // 2:
                    end = start + last_space
                    chunk = text[start:end]
            
            if chunk.strip():
                chunks.append({
                    "text": chunk.strip(),
                    "page": page_num
                })
            
            start = end - overlap
            if start >= len(text):
                break
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def generate_embeddings(chunks, model_name=EMBEDDING_MODEL):
    """Generate embeddings for all chunks."""
    print(f"🤖 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [chunk["text"] for chunk in chunks]
    print(f"⚡ Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings, model

def build_faiss_index(embeddings):
    """Build FAISS index from embeddings."""
    print("🔨 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"✅ Index built with {index.ntotal} vectors")
    return index

def save_index_and_metadata(index, chunks):
    """Save FAISS index and metadata."""
    print(f"💾 Saving index to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)
    
    print(f"💾 Saving metadata to {META_PATH}")
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print("✅ All data saved successfully!")

def main():
    print("=" * 60)
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 60)
    
    # Extract text
    pages = extract_text_from_pdf(PDF_PATH)
    
    # Chunk text
    chunks = chunk_text(pages)
    
    # Generate embeddings
    embeddings, model = generate_embeddings(chunks)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Save everything
    save_index_and_metadata(index, chunks)
    
    print("=" * 60)
    print("🎉 Ingestion complete!")
    print(f"   - Index: {INDEX_PATH}")
    print(f"   - Metadata: {META_PATH}")
    print(f"   - Total chunks: {len(chunks)}")
    print("=" * 60)

if __name__ == "__main__":
    main()

