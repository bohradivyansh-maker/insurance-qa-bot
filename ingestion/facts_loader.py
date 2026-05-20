from pathlib import Path
from langchain_core.documents import Document

FACTS_FILE = Path(__file__).parent.parent / "data" / "policy_facts.txt"

POLICY_NAME_MAP = {
    "CHILDREN": "LIC_ChidrensMoney_BackPlan",
    "ENDOWMENT": "LIC_Endowment",
    "JEEVAN ANAND": "LIC_Jeevan_Anand",
    "JEEVAN LABH": "LIC_Jeevan_Labh",
    "JEEVAN UMANG": "LIC_Jeevan_Umang",
    "JEEVAN AMAR": "LIC_New_Jeevan_Amar",
    "TECH-TERM": "LIC_Tech-Term",
    "TECH TERM": "LIC_Tech-Term",
}

def detect_policy_from_header(header: str) -> str:
    header_upper = header.upper()
    for keyword, filename in POLICY_NAME_MAP.items():
        if keyword in header_upper:
            return filename
    return None

def load_policy_facts():
    with open(FACTS_FILE, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    documents = []
    current_policy_header = None
    current_policy_content = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Detect separator line — 10 or more equals signs
        if len(line) >= 10 and all(c == "=" for c in line):
            # Save previous section
            if current_policy_header and current_policy_content:
                policy_source = detect_policy_from_header(current_policy_header)
                if policy_source:
                    full_content = f"STRUCTURED POLICY FACTS\nPolicy: {current_policy_header}\n\n" + "\n".join(current_policy_content)
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "source": f"D:\\dsw\\Insuarance_qa\\data\\{policy_source}.pdf",
                            "page": 0,
                            "type": "structured_facts"
                        }
                    )
                    documents.append(doc)
                    print(f"Loaded facts for: {current_policy_header} -> {policy_source}")

            # Next line after separator is the policy name
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not all(c == "=" for c in next_line):
                    current_policy_header = next_line
                    current_policy_content = []
                    i += 2  # Skip separator and header
                    # Skip closing separator if present
                    if i < len(lines) and all(c == "=" for c in lines[i].strip()):
                        i += 1
                    continue
        else:
            if current_policy_header and line != current_policy_header:
                current_policy_content.append(lines[i].rstrip())

        i += 1

    # Save last section
    if current_policy_header and current_policy_content:
        policy_source = detect_policy_from_header(current_policy_header)
        if policy_source:
            full_content = f"STRUCTURED POLICY FACTS\nPolicy: {current_policy_header}\n\n" + "\n".join(current_policy_content)
            doc = Document(
                page_content=full_content,
                metadata={
                    "source": f"D:\\dsw\\Insuarance_qa\\data\\{policy_source}.pdf",
                    "page": 0,
                    "type": "structured_facts"
                }
            )
            documents.append(doc)
            print(f"Loaded facts for: {current_policy_header} -> {policy_source}")

    print(f"\nTotal structured policy fact sheets loaded: {len(documents)}")
    return documents

if __name__ == "__main__":
    docs = load_policy_facts()
    for doc in docs:
        print(f"Source: {doc.metadata['source']}")
        print(doc.page_content[:200])
        print("---")