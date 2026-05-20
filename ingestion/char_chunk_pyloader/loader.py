from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = Path(__file__).parent.parent / "data"

def load_documents():
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {DATA_DIR}")

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
        print(f"Loaded {len(docs)} pages from {pdf_path.name}")

    print(f"Total pages loaded: {len(documents)}")
    return documents

if __name__ == "__main__":
    docs = load_documents()