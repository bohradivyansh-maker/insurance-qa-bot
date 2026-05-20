import os
from langfuse import get_client

def get_langfuse():
    return get_client()