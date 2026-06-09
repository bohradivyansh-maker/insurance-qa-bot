# app/langfuse_client.py
# Compatible with Langfuse Python SDK v4
# v4 uses get_client() singleton + propagate_attributes() + start_as_current_observation()

from langfuse.langchain import CallbackHandler
from langfuse import get_client


def get_langfuse_handler() -> CallbackHandler:
    """
    LangChain-compatible callback handler.
    Pass in config={"callbacks": [handler]} on chain.invoke / chain.astream.
    """
    handler = CallbackHandler()
    print("Langfuse CallbackHandler initialized")
    return handler


def get_langfuse_client():
    """
    Returns the global Langfuse v4 client singleton.
    Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL from env.
    Used for manual tracing of guardrail and intent classifier calls.
    """
    client = get_client()
    print("Langfuse client (v4) ready")
    return client