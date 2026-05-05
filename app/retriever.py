import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
from langchain_postgres import PGVector
from embedder import load_embedder

CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg://postgres:Bohra%40123@localhost:5432/insurance_qa"
)

COLLECTION_NAME = "insurance_policies"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25

def get_retriever():
    embedder = load_embedder()
    
    vectorstore = PGVector(
        embeddings=embedder,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True
    )
    return vectorstore

def retrieve_chunks(query: str, vectorstore: PGVector):
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)
    
    filtered = [
        (doc, score) for doc, score in results
        if score >= SIMILARITY_THRESHOLD
    ]
    
    if not filtered:
        return None
    
    print(f"Retrieved {len(filtered)} chunks above threshold {SIMILARITY_THRESHOLD}")
    for i, (doc, score) in enumerate(filtered):
        print(f"Chunk {i+1} | Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")
    
    return filtered

if __name__ == "__main__":
    vectorstore = get_retriever()
    
    test_query = "What is the premium amount for LIC endowment policy?"
    results = vectorstore.similarity_search_with_score(test_query, k=TOP_K)
    for doc, score in results:
        print(f"Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")
    
    if results is None:
        print("No relevant chunks found above similarity threshold")
    else:
        print(f"\nTop chunk content:\n{results[0][0].page_content}")