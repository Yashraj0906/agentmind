import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

REFINER_SYSTEM_PROMPT = """
You are a Query Refinement Agent for AgentMind.

Your job is to take a user query and intent data, then produce 
optimized search queries.

Rules:
- Make queries specific and searchable
- Add current year/month context for current timeframe queries
- For market queries: include ticker symbols if known
- For news queries: add "latest" or "2026" for recency
- For comparison queries: structure for clear comparison search

Return ONLY this JSON, nothing else:
{
    "primary_query": "main optimized search query",
    "secondary_query": "alternative search angle",
    "keywords": ["key", "terms", "list"],
    "search_type": "news/stock/web/none"
}
"""

# **What this does:**
# ```
# primary_query   → main thing to search for
# secondary_query → backup search if first returns bad results
# keywords        → extracted key terms for filtering
# search_type     → tells Data Fetcher which API to use

def refine_query(user_query:str, intent_data:dict) -> dict:
    """
    Takes user query + intent, returns optimized search queries
    """

    context = f"""
    User Query: {user_query}
    Query Type: {intent_data.get('query_type')}
    Domain: {intent_data.get('domain')}
    Timeframe: {intent_data.get('timeframe')}
    Requires Search: {intent_data.get('requires_search')}
    Requires Stock Data: {intent_data.get('requires_stock_data')}
    Current Date: March 2026
    """

    messages = [
        SystemMessage(content=REFINER_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]

    response = llm.invoke(messages)
    raw_output = response.content.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]

    import json
    refined_data = json.loads(raw_output)
    return refined_data

# **Key thing to understand here:**
# ```
# We pass BOTH the user query AND the intent data.
# The refiner now has full context:
# - What the user asked
# - What type of query it is
# - What timeframe
# - What APIs will be needed

# This is agents PASSING INFORMATION to each other.
# This is the core pattern of multi-agent systems.


if __name__ == "__main__":
    from intent_agent import detect_intent
    
    test_queries = [
        "whats happening with nvidia",
        "explain transformers in AI",
        "latest news on OpenAI",
        "compare python vs javascript"
    ]
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        
        intent = detect_intent(query)
        print(f"Intent: {intent['query_type']} | {intent['domain']}")
        
        refined = refine_query(query, intent)
        print(f"Primary Search: {refined['primary_query']}")
        print(f"Secondary Search: {refined['secondary_query']}")
        print(f"Search Type: {refined['search_type']}")
        print("-" * 50)

# **What's new here:**
# ```
# from intent_agent import detect_intent
# → We're importing and USING the previous agent
# → This is the chain forming:
#   Intent Agent → Query Refiner
# → Each agent uses output of previous one