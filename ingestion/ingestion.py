import os
import time
import pymupdf4llm
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# --- CORRECTED IMPORT ---
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING", "postgresql+psycopg2://user:password@localhost:5432/db")
COLLECTION_NAME = "insurance_policies"
PDF_DIRECTORY = "../data/" # Replace with your actual PDF folder path

def wipe_existing_collection(connection_string: str, collection_name: str):
    """Wipes the existing pgVector collection to start fresh."""
    print(f"🧹 Wiping existing collection '{collection_name}'...")
    engine = create_engine(connection_string)
    
    # LangChain-Postgres manages collections via the 'langchain_pg_collection' and 'langchain_pg_embedding' tables.
    # PGVector.drop_collection() handles this safely.
    PGVector.drop_collection(
        engine=engine,
        collection_name=collection_name
    )
    print("✅ Old collection wiped successfully.\n")

def load_and_parse_to_markdown(pdf_path: str):
    """Converts a PDF directly to a Markdown string, preserving tables as Markdown grids."""
    print(f"📄 Parsing {os.path.basename(pdf_path)} to Markdown...")
    # pymupdf4llm automatically detects tables and converts them to markdown formats
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text

def chunk_markdown(md_text: str, source_filename: str):
    """Chunks markdown by headers, keeping tables and sections perfectly intact."""
    # Step 1: Split by headers to keep sections together (Parent chunking concept)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = markdown_splitter.split_text(md_text)

    # Step 2: Ensure no single section is overly massive for the embedding window
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", "|", " ", ""] # Respect markdown tables (|)
    )
    
    final_chunks = text_splitter.split_documents(header_splits)
    
    # Inject Metadata
    for chunk in final_chunks:
        chunk.metadata["source"] = source_filename
        chunk.metadata["is_table"] = "|" in chunk.page_content # Flag for the retriever
        
    return final_chunks

def get_gpu_embeddings():
    """Loads BAAI embeddings directly onto the RTX 3050 GPU using CUDA."""
    print("🧠 Loading BAAI/bge-large-en-v1.5 onto RTX 3050 GPU (CUDA)...")
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'} # Forces the model to use your RTX 3050
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def main():
    start_time = time.time()
    
    # 1. Wipe the old database
    # wipe_existing_collection(POSTGRES_CONNECTION_STRING, COLLECTION_NAME)
    
    # 2. Get GPU Embeddings
    embeddings = get_gpu_embeddings()
    
    # 3. Process all PDFs
    all_chunks = []
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDFs found in {PDF_DIRECTORY}. Please check your path.")
        return

    for pdf_file in pdf_files:
        full_path = os.path.join(PDF_DIRECTORY, pdf_file)
        
        # Parse to Markdown
        md_text = load_and_parse_to_markdown(full_path)
        
        # Chunk intelligently
        chunks = chunk_markdown(md_text, pdf_file)
        all_chunks.extend(chunks)
        
    print(f"\n✂️ Total chunks generated: {len(all_chunks)} (Tables Preserved)")

    # 4. Store in pgVector
    print(f"💾 Pushing chunks and GPU-generated embeddings to pgVector...")
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=all_chunks,
        collection_name=COLLECTION_NAME,
        connection=POSTGRES_CONNECTION_STRING,
        use_jsonb=True,
        pre_delete_collection=True
    )
    
    end_time = time.time()
    print(f"🚀 Ingestion Complete in {round(end_time - start_time, 2)} seconds!")

if __name__ == "__main__":
    main()