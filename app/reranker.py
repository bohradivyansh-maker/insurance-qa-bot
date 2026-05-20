from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N = 7

def load_reranker():
    reranker = CrossEncoder(RERANKER_MODEL)
    print(f"Reranker model loaded: {RERANKER_MODEL}")
    return reranker

def rerank_chunks(query: str, retrieved_chunks: list, reranker: CrossEncoder):
    if retrieved_chunks is None or len(retrieved_chunks) == 0:
        return None
    
    docs = [doc for doc, score in retrieved_chunks]
    
    pairs = [(query, doc.page_content) for doc in docs]
    
    scores = reranker.predict(pairs)
    
    scored_chunks = list(zip(docs, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    top_chunks = scored_chunks[:TOP_N]
    
    print(f"Reranked {len(docs)} chunks, keeping top {TOP_N}")
    for i, (doc, score) in enumerate(top_chunks):
        print(f"Rank {i+1} | Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")
    
    return top_chunks

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from retriever import get_retriever, retrieve_chunks
    
    vectorstore = get_retriever()
    reranker = load_reranker()
    
    test_query = "What is the premium amount for LIC endowment policy?"
    retrieved = retrieve_chunks(test_query, vectorstore)
    
    if retrieved is None:
        print("No chunks retrieved above threshold")
    else:
        reranked = rerank_chunks(test_query, retrieved, reranker)
        print(f"\nTop chunk after reranking:\n{reranked[0][0].page_content}")