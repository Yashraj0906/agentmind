import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage  ##SystemMessage  → the "briefing" you give the LLM :: HumanMessage   → the actual user query


# Load Environment & Initialize LLM

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

#temperature=0     → makes output consistent, not random
#                   (0 = focused, 1 = creative)


INTENT_SYSTEM_PROMPT = """
You are an Intent Classification Agent for AgentMind.

Your ONLY job is to analyze the user query and return a JSON object.

Classify into these query types:
- market     : stocks, crypto, trading, financial markets
- news       : current events, latest updates, what's happening
- technology : tech products, software, AI, programming concepts  
- general    : explanations, definitions, how things work
- comparison : comparing two or more things

Return ONLY this JSON, nothing else, no explanation:
{
    "query_type": "market/news/technology/general/comparison",
    "domain": "main topic area",
    "requires_search": true/false,
    "requires_stock_data": true/false,
    "timeframe": "current/historical/timeless",
    "confidence": "high/medium/low"
}
"""
# **What this does:**
# ```
# This is your "briefing" to the LLM.
# You're telling it:
# - What its job is
# - What categories exist  
# - Exactly what format to return

# temperature=0 + strict JSON prompt = 
# consistent, parseable output every time


## The Main Function
def detect_intent(user_query: str) -> dict:
    """
    Takes user query, returns classified intent as dictionary
    """
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]

    response = llm.invoke(messages)

    raw_output = response.content.strip()

    # clean respone if LLM adds markdown code blocks
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    import json
    intent_data = json.loads(raw_output)

    return intent_data

# **What each part does:**
# ```
# messages = [...]        → packages system + user message together
# llm.invoke(messages)    → sends to Groq, gets response back
# response.content.strip()→ extracts just the text, removes whitespace
# if raw_output...        → sometimes LLM wraps JSON in ```json blocks
#                           this cleans that out
# json.loads(raw_output)  → converts JSON string → Python dictionary
# return intent_data      → sends result to whoever called this function

# Test runner
if __name__ == "__main__":

    test_queries = [
        "What's happening with NVIDIA stock today?",
        "Explain how transformers work in AI",
        "Latest news on OpenAI",
        "Compare Python vs JavaScript for backend"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("Intent:", detect_intent(query))
        print("-" * 50)


# **What this does:**
# ```
# if __name__ == "__main__"  → only runs when YOU run this file directly
#                              won't run when imported by other agents
# test_queries               → 4 different query types to test all paths
# for loop                   → tests each one and prints result