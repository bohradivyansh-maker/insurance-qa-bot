# chain.py
from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from prompts import get_rag_prompt

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

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
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 'unknown')
        context_parts.append(
            f"[Source {i+1}: {source} | Page {page}]\n{doc.page_content}"
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