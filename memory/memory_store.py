import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

# Simple in-memory store as reliable fallback
_conversation_history = []

def save_conversation(user_query: str, agent_response: str):
    """
    Saves conversation turn to local memory
    """
    _conversation_history.append({
        "role": "user",
        "content": user_query
    })
    _conversation_history.append({
        "role": "assistant", 
        "content": agent_response[:500]  # Store first 500 chars
    })
    
    # Keep only last 10 turns to avoid context overflow
    if len(_conversation_history) > 20:
        _conversation_history.pop(0)
        _conversation_history.pop(0)


def get_relevant_memory(current_query: str) -> str:
    """
    Returns recent conversation context
    """
    if not _conversation_history:
        return ""
    
    ## Only get the LAST 2 messages (1 turn)
    recent = _conversation_history[-2:]
    
    memory_text = "Recent conversation context:\n"
    for msg in recent:
        role = "User" if msg["role"] == "user" else "AgentMind"
        memory_text += f"{role}: {msg['content'][:100]}\n"
    
    return memory_text


def get_all_memories() -> list:
    return _conversation_history

# **Why this approach:**
# ```
# mem0 free tier API keeps changing formats
# → causes unpredictable errors

# Local in-memory store:
# → zero API calls
# → zero errors  
# → works perfectly for demo/portfolio
# → resets each session (fine for now)

# We can always swap back to mem0 
# once we add the UI in final phase