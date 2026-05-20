# ingestion/main.py
# Cleaned pipeline — fact_loader removed since fitz now extracts numbers/tables directly.
# Pipeline: load → chunk → embed → store
# Run once during setup or whenever documents are added/changed.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loader import load_documents
from chunker import chunk_documents
from embedder import load_embedder
from store import store_documents

def run_ingestion():
    print("=== Starting Ingestion Pipeline ===")
    print("Loader  : PyMuPDF (fitz) — table and number aware")
    print("Chunker : SemanticChunker — meaning-boundary splitting")
    print("Embedder: BAAI/bge-large-en-v1.5")
    print("Store   : pgVector\n")

    print("[1/4] Loading documents with fitz...")
    documents = load_documents()

    print(f"\n[2/4] Chunking {len(documents)} pages semantically...")
    chunks = chunk_documents(documents)

    print(f"\n[3/4] Loading embedding model...")
    embedder = load_embedder()

    print(f"\n[4/4] Storing {len(chunks)} chunks in pgVector...")
    store_documents(chunks, embedder)

    print("\n=== Ingestion Complete ===")
    print(f"Total chunks stored: {len(chunks)}")
    print("\nNote: Old chunks from previous ingestion runs are replaced.")
    print("If results are worse than before, restore original loader.py")
    print("and chunker.py from backup and re-run ingestion.")

if __name__ == "__main__":
    run_ingestion()