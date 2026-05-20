# ingestion/loader.py
# Uses PyMuPDF (fitz) in "dict" mode with spatial block merging.
#
# WHY THIS APPROACH:
# get_text("text") outputs each table cell as a separate line (e.g. "Rs. 15,000/-"
# is 12 chars). chunker.py's < 80 char filter then drops every table row, so
# instalment amounts, loan percentages, and any other tabular data never reaches
# pgvector and the model cannot answer those queries.
#
# get_text("dict") gives per-line bounding boxes. We reconstruct blocks by joining
# lines within a block into one string, then spatially merging blocks that share
# vertical space (i.e. columns in the same table row). The result: every table
# row becomes one long string that survives the 80-char filter intact.
#
# PLAN TYPE FIX:
# The plan type line e.g. "(A Par, Non-Linked, Life, Individual, Savings Plan)"
# sits in a formatted header block on page 0 of every policy PDF. dict-mode
# spatial merging drops this line because it gets merged into surrounding blocks
# and loses position priority. We extract it separately from raw get_text("text")
# for page 0 only and inject it at the top of the chunk so it is always present
# in the vector store and retrievable for plan type queries.

import re
from pathlib import Path
import fitz  # PyMuPDF
from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent.parent / "data"

# Matches plan type lines like:
# (A Par, Non-Linked, Life, Individual, Savings Plan)
# (A Non-Par, Non-Linked, Life, Individual, Pure Risk Plan)
# (A Par, Non-Linked, Individual, Savings, Whole Life Insurance plan)
PLAN_TYPE_PATTERN = re.compile(
    r'\(A\s+(?:Par|Non-Par)[^\)]{5,80}\)',
    re.IGNORECASE
)


def _blocks_overlap_vertically(bbox1, bbox2, tolerance: int = 5) -> bool:
    """
    Returns True if two bounding boxes share vertical space.
    Used to detect table columns that sit side-by-side on the same row.
    tolerance=5 points handles minor PDF alignment differences.
    """
    _, y1_top, _, y1_bot = bbox1
    _, y2_top, _, y2_bot = bbox2
    return not (y1_bot + tolerance < y2_top or y2_bot + tolerance < y1_top)


def _extract_plan_type(page) -> str | None:
    """
    Extracts the plan type line from the first 10 lines of raw text on page 0.
    Uses get_text("text") which reads strictly top-to-bottom and always captures
    the header block that dict-mode spatial merging can drop.
    Returns the plan type string e.g. "(A Par, Non-Linked, Life, Individual, Savings Plan)"
    or None if not found.
    """
    raw = page.get_text("text").strip()
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    for line in lines[:10]:
        if PLAN_TYPE_PATTERN.search(line):
            return line.strip()
    return None


def _extract_page_text(page, page_num: int = -1) -> str:
    """
    Extracts text from a single fitz page using dict mode.

    Steps:
    1. Pull all text blocks with per-line bounding boxes.
    2. Within each block, join lines into one string separated by ' | '.
       This keeps table headers and values on one line instead of separate tiny lines.
    3. Merge blocks that overlap vertically — these are multi-column table cells
       that fitz splits into separate blocks because they sit in different columns.
    4. For page 0 only: inject the plan type line at the top if dict-mode dropped it.
    4. Return reconstructed blocks joined by double newlines.
    """
    page_dict = page.get_text("dict")

    # Step 1 & 2: extract blocks, join lines within each block
    raw_blocks = []
    for block in page_dict["blocks"]:
        if block["type"] != 0:  # skip image blocks
            continue
        line_texts = []
        for line in block["lines"]:
            line_text = " ".join(span["text"] for span in line["spans"]).strip()
            if line_text:
                line_texts.append(line_text)
        if line_texts:
            raw_blocks.append((block["bbox"], " | ".join(line_texts)))

    # Step 3: merge vertically-overlapping blocks (multi-column table rows)
    merged = []
    used = set()

    for i, (bbox_i, text_i) in enumerate(raw_blocks):
        if i in used:
            continue
        group_text = text_i
        group_bbox = list(bbox_i)

        for j, (bbox_j, text_j) in enumerate(raw_blocks):
            if j <= i or j in used:
                continue
            if _blocks_overlap_vertically(bbox_i, bbox_j):
                group_text += " | " + text_j
                group_bbox[0] = min(group_bbox[0], bbox_j[0])
                group_bbox[2] = max(group_bbox[2], bbox_j[2])
                group_bbox[3] = max(group_bbox[3], bbox_j[3])
                used.add(j)

        used.add(i)
        merged.append(group_text)

    result = "\n\n".join(merged)

    # Step 4: for page 0, inject plan type at top if dict-mode dropped it
    if page_num == 0:
        plan_type = _extract_plan_type(page)
        if plan_type and plan_type not in result:
            result = plan_type + "\n\n" + result

    return result


def load_documents() -> list:
    """
    Loads all PDFs from data/ directory using PyMuPDF (fitz) dict-mode extraction.

    Each page becomes one LangChain Document with metadata:
      - source     : full path to PDF
      - page       : page number (0-indexed)
      - total_pages: total pages in the document

    Returns list of LangChain Document objects.
    """
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {DATA_DIR}")

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            file_docs = []

            for page_num in range(page_count):
                page = doc[page_num]
                text = _extract_page_text(page, page_num=page_num).strip()

                # Skip near-empty pages (image-only pages return very little text)
                if len(text) < 50:
                    print(f"  Skipping near-empty page {page_num + 1} in {pdf_path.name}")
                    continue

                langchain_doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_num,
                        "total_pages": page_count,
                    },
                )
                file_docs.append(langchain_doc)

            doc.close()
            documents.extend(file_docs)
            print(
                f"Loaded {len(file_docs)} pages from {pdf_path.name} "
                f"(skipped {page_count - len(file_docs)} empty pages)"
            )

        except Exception as e:
            print(f"ERROR loading {pdf_path.name}: {e}")
            continue

    if not documents:
        raise ValueError("No documents loaded — all PDFs may be image-based or empty.")

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    # Quick verification — check plan type extraction for all PDFs
    import fitz
    DATA_DIR_CHECK = Path(__file__).parent.parent / "data"
    print("=== Plan type extraction check ===")
    for pdf_path in sorted(DATA_DIR_CHECK.glob("*.pdf")):
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        plan_type = _extract_plan_type(page)
        print(f"{pdf_path.name}: {plan_type}")
        doc.close()

    print("\n=== Full load ===")
    docs = load_documents()
    if docs:
        print("\n--- Sample page 0 extraction (first PDF) ---")
        print(docs[0].page_content[:400])
        print(f"\nMetadata: {docs[0].metadata}")