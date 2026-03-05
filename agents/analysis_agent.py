import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from agents.models import FetchedData, AnalysisResult

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


ANALYSIS_SYSTEM_PROMPT = """
You are an Analysis Agent for AgentMind.

You receive raw search results and must extract real intelligence from them.

Your job:
1. Read all provided content carefully
2. Extract the most important insights
3. Remove noise and irrelevant information
4. Assess overall sentiment
5. Suggest follow-up questions

Return ONLY this JSON, nothing else:
{
    "summary": "2-3 sentence summary of everything found",
    "key_insights": [
        "Most important insight with context",
        "Second insight with context", 
        "Third insight with context"
    ],
    "sentiment": "positive/negative/neutral/mixed",
    "confidence": "high/medium/low",
    "sources": ["source title 1", "source title 2"],
    "follow_up_suggestions": [
        "A natural follow-up question",
        "Another angle to explore"
    ]
}
"""

def analyze_data(fetched_data: dict, intent_data: dict) -> AnalysisResult:
    """
    Analyzes fetched results and extracts key insights
    """
    
    # Build context from all results
    results_text = ""
    for i, result in enumerate(fetched_data.get("results", []), 1):
        results_text += f"""
Source {i}: {result.get('title', '')}
Content: {result.get('content', '')[:500]}
URL: {result.get('url', '')}
---
"""
    
    context = f"""
Original Query: {fetched_data.get('query')}
Query Type: {intent_data.get('query_type')}
Domain: {intent_data.get('domain')}

Raw Results to Analyze:
{results_text}

Please analyze these results and extract key insights.
"""
    
    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]
    
    response = llm.invoke(messages)
    raw_output = response.content.strip()
    
    # Clean markdown if present
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
    
    data = json.loads(raw_output)
    
    # Add query to data
    data["query"] = fetched_data.get("query")
    
    # Return as Pydantic model — validated and type safe
    return AnalysisResult(**data)

# **Key concept here:**
# ```
# results_text building loop:
# → Takes all 5 raw Tavily results
# → Combines them into one big text block
# → Sends entire context to LLM at once
# → LLM reads everything and finds patterns

# return AnalysisResult(**data)
# → ** unpacks dict into Pydantic model
# → Pydantic validates every field
# → If anything is wrong, clear error thrown
# → If all good, clean typed object returned

if __name__ == "__main__":
    from agents.intent_agent import detect_intent
    from agents.query_refiner import refine_query
    from agents.data_fetcher import fetch_data
    
    query = "whats happening with nvidia today"
    
    print(f"Query: {query}")
    print("="*50)
    
    intent = detect_intent(query)
    refined = refine_query(query, intent)
    fetched = fetch_data(query, refined, intent)
    analysis = analyze_data(fetched, intent)
    
    print(f"\n📋 SUMMARY:\n{analysis.summary}")
    print(f"\n🔍 KEY INSIGHTS:")
    for i, insight in enumerate(analysis.key_insights, 1):
        print(f"  {i}. {insight}")
    print(f"\n📊 SENTIMENT: {analysis.sentiment}")
    print(f"🎯 CONFIDENCE: {analysis.confidence}")
    print(f"\n💡 FOLLOW-UP QUESTIONS:")
    for q in analysis.follow_up_suggestions:
        print(f"  → {q}")