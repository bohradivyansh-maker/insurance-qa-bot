# insurance_agent/agent/memory.py
# Simple custom memory — avoids langchain.memory version issues entirely

class SimpleMemory:
    """
    Lightweight conversation memory.
    Stores list of {"role": "user"/"assistant", "content": "..."} dicts.
    Stateless across sessions — resets on every new chat start.
    """
    def __init__(self):
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_messages_for_groq(self) -> list:
        """Returns messages in Groq API format."""
        return self.messages

    def clear(self):
        self.messages = []

def create_memory() -> SimpleMemory:
    return SimpleMemory()

def get_history_as_text(memory: SimpleMemory) -> str:
    if not memory.messages:
        return "No prior conversation."
    lines = []
    for msg in memory.messages:
        role = "User" if msg["role"] == "user" else "Advisor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)