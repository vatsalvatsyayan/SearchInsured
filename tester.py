import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# Configuration
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8  # Increased from 4 to get more context

def load_index_and_metadata():
    """Load FAISS index and metadata."""
    print("📂 Loading index and metadata...")
    index = faiss.read_index(INDEX_PATH)
    
    with open(META_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"✅ Loaded {index.ntotal} vectors and {len(chunks)} chunks")
    return index, chunks

def expand_query(query):
    """Expand the user query with related terms for better retrieval."""
    expansions = {
        'pediatrician': 'pediatrician paediatrician child doctor specialist primary care',
        'paediatrician': 'pediatrician paediatrician child doctor specialist primary care',
        'cardiologist': 'cardiologist heart doctor specialist cardiovascular',
        'dermatologist': 'dermatologist skin doctor specialist',
        'orthopedist': 'orthopedist orthopedic bone doctor specialist surgeon',
        'neurologist': 'neurologist brain doctor specialist neurology',
        'psychiatrist': 'psychiatrist mental health doctor specialist therapy',
        'dentist': 'dentist dental orthodontist oral care',
        'therapist': 'therapist therapy mental health counseling psychologist',
        'physical therapy': 'physical therapy PT rehabilitation therapy',
        'prescription': 'prescription medication drug pharmacy',
        'surgery': 'surgery surgical procedure operation hospital',
        'emergency': 'emergency urgent care ER hospital',
        'maternity': 'maternity pregnancy prenatal childbirth delivery',
        'preventive': 'preventive screening wellness checkup annual exam',
    }
    
    query_lower = query.lower()
    expanded = query
    
    for key, expansion in expansions.items():
        if key in query_lower:
            expanded = f"{query} {expansion}"
            break
    
    return expanded

def retrieve_similar_chunks(query, index, chunks, model, k=TOP_K):
    """Retrieve top-k similar chunks for a query."""
    # Expand query for better retrieval
    expanded_query = expand_query(query)
    
    print(f"🔍 Searching for top {k} relevant chunks...")
    if expanded_query != query:
        print(f"   Expanded query with related terms")
    
    query_embedding = model.encode([expanded_query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "distance": float(dist)
        })
    
    return results

def generate_answer(query, retrieved_chunks):
    """Generate answer using Gemini with retrieved context."""
    print("🤖 Generating answer with Gemini...")
    
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Chunk {i}, Page {chunk['page']}]:\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Improved prompt that allows reasoning and inference
    prompt = f"""You are a helpful insurance assistant. Answer the user's question using the provided context from an insurance policy document.

INSTRUCTIONS:
1. Read the context carefully and answer based on the information provided
2. If the exact answer isn't stated but can be reasonably inferred from related information (e.g., "specialist visits" covers "pediatricians"), make that connection and explain it
3. Always cite page numbers using the format (p.X) when referencing specific information
4. If the context mentions general categories (like "specialist visits" or "primary care"), explain how those relate to the user's specific question
5. Be helpful and conversational - interpret the user's intent
6. Only say "I cannot find specific information about this in the provided document" if there is truly no relevant information or reasonable inference possible
7. If coverage depends on conditions (in-network vs out-network, referrals, etc.), mention those conditions

CONTEXT FROM INSURANCE DOCUMENT:
{context}

USER QUESTION: {query}

HELPFUL ANSWER:"""
    
    # Call Gemini API
    api_key = "AIzaSyBcQOTTdLrLTuMZo9tJ96IPNmE3dhoU7nI"
    if not api_key:
        return "❌ ERROR: GEMINI_API_KEY environment variable not set"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ ERROR calling Gemini API: {str(e)}"

def main():
    print("=" * 60)
    print("🤖 RAG Insurance Assistant - CLI Tester")
    print("=" * 60)
    
    # Load resources
    index, chunks = load_index_and_metadata()
    print("🤖 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded")
    
    print("\n" + "=" * 60)
    print("Ready! Type your question (or 'quit' to exit)")
    print("=" * 60 + "\n")
    
    while True:
        query = input("❓ Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        print()
        
        # Retrieve similar chunks
        retrieved = retrieve_similar_chunks(query, index, chunks, model)
        
        # Show retrieved chunks (abbreviated)
        print("\n📄 Retrieved Context Summary:")
        for i, chunk in enumerate(retrieved[:4], 1):  # Show top 4
            print(f"  [{i}] Page {chunk['page']} (distance: {chunk['distance']:.4f})")
            preview = chunk['text'][:100].replace('\n', ' ') + "..."
            print(f"      {preview}")
        if len(retrieved) > 4:
            print(f"  ... and {len(retrieved) - 4} more chunks")
        
        # Generate answer
        answer = generate_answer(query, retrieved)
        
        print("\n" + "=" * 60)
        print("💬 ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()