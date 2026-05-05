# app/main.py
import os
import chainlit as cl
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from retriever import get_retriever, retrieve_chunks
from reranker import load_reranker, rerank_chunks
from chain import load_llm, run_chain, format_context
from langfuse_client import get_langfuse_handler
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

GROQ_MODEL = "llama-3.3-70b-versatile"

GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for an Insurance Policy Assistant.
Your only job is to classify if the user query is related to insurance or not.
Reply with exactly one word: YES or NO.
YES = query is about insurance, policies, premiums, claims, coverage, IRDAI, or any insurance topic.
YES = query is a greeting like hello, hi, thanks, bye, how are you.
NO = query is about anything else completely unrelated to insurance."""),
    ("human", "{query}")
])

async def check_insurance_guardrail(query: str, llm) -> bool:
    chain = GUARDRAIL_PROMPT | llm
    response = await chain.ainvoke({"query": query})
    result = response.content.strip().upper()
    return result == "YES"

@cl.on_chat_start
async def on_chat_start():
    vectorstore = get_retriever()
    reranker = load_reranker()
    llm = load_llm()
    langfuse_handler = get_langfuse_handler()

    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("reranker", reranker)
    cl.user_session.set("llm", llm)
    cl.user_session.set("langfuse_handler", langfuse_handler)

    await cl.Message(
        content="👋 Welcome to the Insurance Policy Assistant!\n\nI can answer questions about LIC, HDFC Ergo, Star Health policies and IRDAI guidelines.\n\nAsk me anything about your insurance policies."
    ).send()

GREETING_RESPONSES = {
    "hello": "👋 Hello! I am your Insurance Policy Assistant. Ask me anything about LIC policies or IRDAI guidelines.",
    "hi": "👋 Hi there! How can I help you with your insurance queries today?",
    "hey": "👋 Hey! Ask me anything about LIC policies or IRDAI guidelines.",
    "thanks": "😊 No problem! Glad I could help. Feel free to ask anything else.",
    "thank you": "😊 Happy to help! Let me know if you have more questions.",
    "bye": "👋 Goodbye! Stay insured and stay protected!",
    "goodbye": "👋 Take care! Come back anytime you have insurance questions.",
    "how are you": "😊 I am doing great and ready to help! Ask me anything about your insurance policies.",
}

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content
    query_lower = query.strip().lower()
    
    if query_lower in GREETING_RESPONSES:
        await cl.Message(content=GREETING_RESPONSES[query_lower]).send()
        return

    vectorstore = cl.user_session.get("vectorstore")
    reranker = cl.user_session.get("reranker")
    llm = cl.user_session.get("llm")
    langfuse_handler = cl.user_session.get("langfuse_handler")

    is_insurance = await check_insurance_guardrail(query, llm)
    if not is_insurance:
        await cl.Message(
            content="⚠️ I am designed to answer insurance-related questions only. Please ask me about insurance policies, premiums, claims, or coverage."
        ).send()
        return

    retrieved = retrieve_chunks(query, vectorstore)
    reranked = rerank_chunks(query, retrieved, reranker)

    msg = cl.Message(content="")
    await msg.send()

    if reranked is None:
        msg.content = "I cannot find specific information about this in the available policy documents."
        await msg.update()
        return

    context = format_context(reranked)
    from prompts import get_rag_prompt
    prompt = get_rag_prompt()
    chain = prompt | llm

    async for chunk in chain.astream(
        {"context": context, "question": query},
        config={"callbacks": [langfuse_handler]}
    ):
        await msg.stream_token(chunk.content)

    await msg.update()