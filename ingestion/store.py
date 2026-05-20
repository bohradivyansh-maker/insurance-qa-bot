from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from embedder import load_embedder
import os

CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg://postgres:Bohra%40123@localhost:5432/insurance_qa"
)

COLLECTION_NAME = "insurance_policies"

def get_vectorstore(embedder):
    vectorstore = PGVector(
        embeddings=embedder,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
        pre_delete_collection=True
    )
    return vectorstore

def store_documents(chunks, embedder):
    vectorstore = get_vectorstore(embedder)
    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks in pgVector")
    return vectorstore

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from loader import load_documents
    from chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    embedder = load_embedder()
    store_documents(chunks, embedder)
    