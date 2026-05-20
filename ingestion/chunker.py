# ingestion/chunker.py
# Replaced RecursiveCharacterTextSplitter with LangChain SemanticChunker.
# SemanticChunker splits on meaning boundaries using the same BAAI embedder,
# so each chunk contains one complete idea rather than an arbitrary text window.
# Section-aware heading prefix is preserved from original implementation.

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_experimental.text_splitter import SemanticChunker
from embedder import load_embedder

# Heading pattern — detects LIC document section headers like:
# "3. Death Benefit:", "8. Grace Period:", "12. Maturity Benefit:"
HEADING_PATTERN = re.compile(
    r'^(\d+[\.\)]\s+[A-Z][^\n]{3,60}[:–-])',
    re.MULTILINE
)

def detect_section_heading(text: str) -> str | None:
    """
    Extracts the first section heading found in a chunk.
    Used to prefix chunks with their parent section for retrieval context.
    """
    match = HEADING_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None

def chunk_documents(documents: list) -> list:
    """
    Semantically chunks all loaded documents using SemanticChunker.

    SemanticChunker works by:
    1. Splitting text into sentences
    2. Embedding each sentence using the same BAAI/bge-large-en-v1.5 embedder
    3. Finding points where embedding similarity drops sharply (topic shift)
    4. Cutting at those boundaries

    Result: chunks align with natural section boundaries in LIC documents
    rather than arbitrary 1000-character windows.

    breakpoint_threshold_type options:
      - "percentile"    : cuts where similarity drops below Nth percentile (default)
      - "standard_deviation" : cuts at points more than N std devs below mean
      - "interquartile" : uses IQR method

    We use "percentile" with threshold=85 — cuts at the sharpest 15% of
    topic shifts. Higher = fewer, larger chunks. Lower = more, smaller chunks.
    Tune this if chunks are too large or too small after testing.
    """
    print("Loading embedder for SemanticChunker...")
    embedder = load_embedder()

    splitter = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

    all_chunks = []
    skipped    = 0

    for doc in documents:
        source   = doc.metadata.get("source", "unknown")
        page_num = doc.metadata.get("page", 0)

        # Get filename for section prefix
        source_name = os.path.basename(source)

        # SemanticChunker splits one document at a time
        try:
            chunks = splitter.create_documents(
                texts=[doc.page_content],
                metadatas=[doc.metadata]
            )
        except Exception as e:
            print(f"  WARNING: SemanticChunker failed on page {page_num} of {source_name}: {e}")
            print(f"  Falling back to full page as single chunk.")
            chunks = [doc]

        for chunk in chunks:
            # Skip near-empty chunks (fitz image-only page remnants)
            if len(chunk.page_content.strip()) < 80:
                skipped += 1
                continue

            # Section-aware prefix — prepend heading to chunk for retrieval context
            heading = detect_section_heading(chunk.page_content)
            if heading:
                chunk.page_content = (
                    f"[Document: {source_name} | Section: {heading}]\n\n"
                    + chunk.page_content
                )
            else:
                chunk.page_content = (
                    f"[Document: {source_name}]\n\n"
                    + chunk.page_content
                )

            all_chunks.append(chunk)

    print(f"\nTotal chunks created : {len(all_chunks)}")
    print(f"Near-empty chunks skipped: {skipped}")

    if all_chunks:
        print(f"\nSample chunk (first):")
        print(all_chunks[0].page_content[:300])
        print(f"Metadata: {all_chunks[0].metadata}")

    return all_chunks


if __name__ == "__main__":
    from loader import load_documents
    docs   = load_documents()
    chunks = chunk_documents(docs)
    print(f"\nFinal chunk count: {len(chunks)}")