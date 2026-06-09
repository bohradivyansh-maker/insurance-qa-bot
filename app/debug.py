import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING", "postgresql+psycopg2://user:password@localhost:5432/db")
COLLECTION_NAME = "insurance_policies"

print("🧠 Loading BAAI/bge-large-en-v1.5 onto RTX 3050 GPU (CUDA)...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print(f"🔌 Connecting to pgVector collection: '{COLLECTION_NAME}'...")
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_CONNECTION_STRING,
    use_jsonb=True
)

# Initialize Groq with temperature=0 for strict data extraction
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# The new Strict Compliance Prompt
prompt_template = ChatPromptTemplate.from_template("""
You are a strict data extractor for an insurance company. 
Quote the exact numbers from the context below. Do not round, summarize, or estimate numbers.
If the exact value is not present in the text or tables below, you MUST reply: 'The policy document does not explicitly state this value.'

Context:
{context}

Question:
{question}

Answer:
""")

def test_query(query: str):
    print(f"\n🔍 Searching for: '{query}'")
    # Fetch top 3 chunks
    docs = vectorstore.similarity_search_with_score(query, k=3)

    if not docs:
        print("❌ No relevant chunks found.")
        return

    print("\n" + "="*60)
    print(" 📑 TOP RETRIEVED CHUNKS (Verify Markdown Tables Here)")
    print("="*60)
    
    context_parts = []
    for i, (doc, score) in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        is_table = doc.metadata.get('is_table', False)
        print(f"\n[Chunk {i} | Source: {source} | Has Table: {is_table} | Vector Distance: {round(score, 4)}]")
        print("-" * 60)
        # Print up to 800 characters so you can see the table structure
        print(doc.page_content[:800] + ("\n...[truncated]" if len(doc.page_content) > 800 else ""))
        context_parts.append(doc.page_content)

    context_str = "\n\n".join(context_parts)

    print("\n" + "="*60)
    print(" 🤖 GROQ LLM ANSWER")
    print("="*60)
    chain = prompt_template | llm
    response = chain.invoke({"context": context_str, "question": query})
    print(response.content)
    print("="*60)

if __name__ == "__main__":
    print("\n✅ Debug Environment Ready!")
    print("Type 'exit' to quit.\n")
    print("💡 Try asking hard numerical/table questions. Examples:")
    print("  - 'What is the surrender value percentage for year 3 in LIC Jeevan Labh?'")
    print("  - 'What is the minimum maturity age for LIC Jeevan Anand?'")
    print("  - 'List the minimum instalment amounts for monthly and yearly modes in Tech Term.'")

    while True:
        q = input("\n🤔 Enter your test query: ")
        if q.lower() in ['exit', 'quit']:
            break
        if q.strip():
            test_query(q)