from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Filter out low-content chunks
    filtered_chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
    
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