from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_embedder():
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"Embedding model loaded: {EMBEDDING_MODEL}")
    return embedder

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from loader import load_documents
    from chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    
    embedder = load_embedder()
    sample_vector = embedder.embed_query(chunks[0].page_content)
    print(f"Embedding dimensions: {len(sample_vector)}")
    print(f"Sample vector (first 5 values): {sample_vector[:5]}")