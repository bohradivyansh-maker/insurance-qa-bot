# check_old_vs_new.py in ingestion/
import fitz
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
pdf_path = DATA_DIR / "LIC_ChidrensMoney_BackPlan.pdf"

doc = fitz.open(str(pdf_path))
page = doc[0]

print("=== get_text('text') output (first 500 chars) ===")
print(page.get_text("text").strip()[:500])