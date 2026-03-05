import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from agents.models import AnalysisResult

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# **One change here:**
# ```
# temperature=0.3 instead of 0
# → Slight creativity allowed for natural language
# → Makes responses feel human, not robotic
# → Still controlled, not random

RESPONSE_SYSTEM_PROMPT = """
You are the Response Agent for AgentMind — a universal intelligence system.

Your job is to take analyzed data and write a clear, helpful response.

Rules:
- Write in a clean, professional but conversational tone
- Use the exact data provided, don't add information
- Structure response with clear sections
- Always end with follow-up suggestions
- Keep it concise but complete

Format your response exactly like this:

## 🔍 {topic}

**Summary**
{2-3 sentence summary}

**Key Insights**
- {insight 1}
- {insight 2}  
- {insight 3}

**Sentiment:** {sentiment} | **Confidence:** {confidence}

**You might also want to ask:**
→ {follow up 1}
→ {follow up 2}

---
*Powered by AgentMind*
"""

def generate_response(
    user_query: str,
    analysis: AnalysisResult,
    intent_data: dict
) -> str:
    """
    Takes analysis result, generates final human-readable response
    """
    
    context = f"""
User asked: {user_query}
Query type: {intent_data.get('query_type')}

Analysis Data:
- Summary: {analysis.summary}
- Key Insights: {analysis.key_insights}
- Sentiment: {analysis.sentiment}
- Confidence: {analysis.confidence}
- Sources: {analysis.sources}
- Follow-up suggestions: {analysis.follow_up_suggestions}

Write a clean, formatted response using this data.
"""
    
    messages = [
        SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]
    
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    from agents.intent_agent import detect_intent
    from agents.query_refiner import refine_query
    from agents.data_fetcher import fetch_data
    from agents.analysis_agent import analyze_data
    
    test_queries = [
        "whats happening with nvidia today",
        "explain how attention mechanism works in transformers"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"USER: {query}")
        print('='*60)
        
        intent = detect_intent(query)
        refined = refine_query(query, intent)
        fetched = fetch_data(query, refined, intent)
        analysis = analyze_data(fetched, intent)
        response = generate_response(query, analysis, intent)
        
        print(response)