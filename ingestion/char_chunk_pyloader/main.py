import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loader import load_documents
from chunker import chunk_documents
from embedder import load_embedder
from store import store_documents, get_vectorstore
from facts_loader import load_policy_facts

def run_ingestion():
    print("=== Starting Ingestion Pipeline ===")

    print("\n[1/5] Loading documents...")
    documents = load_documents()

    print("\n[2/5] Loading structured policy facts...")
    facts = load_policy_facts()
    documents = documents + facts

    print("\n[3/5] Chunking documents...")
    chunks = chunk_documents(documents)

    print("\n[4/5] Loading embedding model...")
    embedder = load_embedder()

    print("\n[5/5] Storing chunks in pgVector...")
    store_documents(chunks, embedder)

    print("\n=== Ingestion Complete ===")
    print(f"Total chunks stored: {len(chunks)}")

if __name__ == "__main__":
    run_ingestion()