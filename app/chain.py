# chain.py
from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from prompts import get_rag_prompt

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Query expansion map — when a query contains the key phrase, the value terms
# are appended to the retrieval query to surface the right chunks.
# The original user question is kept unchanged for the LLM answer.
QUERY_EXPANSIONS = {
    "maturity benefit": "sum assured on maturity basic sum assured payable on survival",
    "entry age": "minimum age at entry years completed eligible life assured",
    "minor": "minimum age at entry years completed life assured",
    "proposer dies": "premium waiver benefit rider proposer death future premiums waived",
    "proposer death": "premium waiver benefit rider proposer death future premiums waived",
    "suicide": "suicide exclusion 12 months commencement revival nominee payable",
    "surrender calculat": "surrender value formula guaranteed special unexpired risk premium",
    "foreclosure": "loan interest default surrender value outstanding foreclosure",
    "settlement option": "maturity benefit instalments 5 10 15 years settlement option",
    "survival benefit": "survival benefit percentage money back periodic payout",
    "what type of plan": "par non-linked non par life individual savings pure risk whole life participating",
    "type of plan": "par non-linked non par life individual savings pure risk whole life participating"
}

def expand_query(query: str) -> str:
    """
    Enriches the retrieval query with section-specific keywords.
    Only used for vector/BM25 retrieval — the original query is
    always passed to the LLM unchanged so the answer stays on-topic.
    """
    query_lower = query.lower()
    expansions = []
    for trigger, addition in QUERY_EXPANSIONS.items():
        if trigger in query_lower:
            expansions.append(addition)
    if expansions:
        expanded = query + " " + " ".join(expansions)
        print(f"Query expanded for retrieval: {expanded[:120]}")
        return expanded
    return query

def load_llm():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
        streaming=True
    )
    print(f"LLM loaded: {GROQ_MODEL}")
    return llm

def format_context(reranked_chunks):
    context_parts = []
    for i, (doc, score) in enumerate(reranked_chunks):
        context_parts.append(
            f"[Context {i+1}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)

def run_chain(query: str, reranked_chunks, llm, langfuse_handler=None):
    if reranked_chunks is None:
        return "I cannot find specific information about this in the available policy documents."

    context = format_context(reranked_chunks)
    prompt = get_rag_prompt()
    chain = prompt | llm

    config = {}
    if langfuse_handler:
        config = {"callbacks": [langfuse_handler]}

    response = chain.invoke(
        {"context": context, "question": query},
        config=config
    )

    return response.content

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from retriever import get_retriever, retrieve_chunks
    from reranker import load_reranker, rerank_chunks
    from langfuse_client import get_langfuse_handler

    vectorstore = get_retriever()
    reranker = load_reranker()
    llm = load_llm()
    langfuse_handler = get_langfuse_handler()

    test_query = "What are the instalment payment options available?"

    retrieved = retrieve_chunks(test_query, vectorstore)
    reranked = rerank_chunks(test_query, retrieved, reranker)
    answer = run_chain(test_query, reranked, llm, langfuse_handler)

    print(f"\nQuestion: {test_query}")
    print(f"\nAnswer: {answer}")