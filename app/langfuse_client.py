import os
from langfuse.langchain import CallbackHandler

def get_langfuse_handler():
    handler = CallbackHandler()
    print("Langfuse handler initialized")
    return handler