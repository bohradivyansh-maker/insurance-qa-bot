import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

SECTION_HEADING_PATTERN = re.compile(
    r'^(\d+[\.\)]\s+[A-Z][a-zA-Z\s\(\)\/\-]+:)', 
    re.MULTILINE
)

def extract_section_heading(text: str) -> str:
    match = SECTION_HEADING_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    enhanced_chunks = []
    for chunk in chunks:
        heading = extract_section_heading(chunk.page_content)
        source = chunk.metadata.get("source", "")
        doc_name = source.split("\\")[-1].replace(".pdf", "").replace("_", " ")

        if heading:
            enhanced_content = f"Section: {heading} | Document: {doc_name}\n\n{chunk.page_content}"
        else:
            enhanced_content = f"Document: {doc_name}\n\n{chunk.page_content}"

        enhanced_chunk = Document(
            page_content=enhanced_content,
            metadata=chunk.metadata
        )
        enhanced_chunks.append(enhanced_chunk)

    filtered_chunks = [c for c in enhanced_chunks if len(c.page_content.strip()) > 50]

    print(f"Total chunks before filtering: {len(chunks)}")
    print(f"Total chunks after filtering: {len(filtered_chunks)}")
    return filtered_chunks

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from loader import load_documents
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Sample chunk:\n{chunks[0].page_content}")
    print(f"Metadata: {chunks[0].metadata}")