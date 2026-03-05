import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agents.intent_agent import detect_intent
from agents.query_refiner import refine_query
from agents.data_fetcher import fetch_data
from agents.analysis_agent import analyze_data
from agents.response_agent import generate_response

load_dotenv()

# **What's new:**
# ```
# TypedDict    → defines the shape of our shared state
# StateGraph   → LangGraph's main class for building graphs
# END          → special LangGraph marker meaning "pipeline done"


## Defined Shared State
class AgentState(TypedDict):
    """
    This is the SHARED MEMORY of the entire pipeline.
    Every agent reads from and writes to the state.
    Think of it as a shared whiteboard all agents can see.
    """
    user_query: str
    memory_context: str
    intent: dict
    refined: dict
    fetched: dict
    analysis: dict
    response: str
    error: str


# **This is the most important concept in LangGraph:**
# ```
# In our old main.py:
# → each agent returned a value
# → we manually passed it to next agent

# In LangGraph:
# → there is ONE shared state object
# → each agent READS what it needs from state
# → each agent WRITES its output back to state
# → next agent automatically has access

# Like a relay race where the baton 
# contains ALL previous runners' notes


###  Define Each Node (Agent Wrapper)

def memory_node(state: AgentState) -> AgentState:
    """Node 0 — Fetch relevant memory before processing"""
    try:
        from memory.memory_store import get_relevant_memory
        memory_context = get_relevant_memory(state["user_query"])
        return {"memory_context": memory_context}
    except Exception as e:
        return {"memory_context": ""}

def intent_node(state: AgentState) -> AgentState:
    """Node 1 — Detect intent with focused memory context"""
    try:
        query = state["user_query"]
        memory = state.get("memory_context", "")
        
        # Only inject memory for short/vague follow-up queries
        vague_words = ["it", "they", "their", "this", "that", "he", "she"]
        is_followup = any(word in query.lower().split() for word in vague_words)
        
        if memory and is_followup:
            query = f"{query} (context: {memory[:150]})"
        
        intent = detect_intent(query)
        return {"intent": intent}
    except Exception as e:
        return {"error": f"Intent detection failed: {e}"}

# **What this fixes:**
# ```
# Before: always injected full memory → confused the LLM
# After:  only injects memory when query contains
#         vague words like "it", "they", "their"
#         AND keeps it short (150 chars max)

# "How does it compare to AMD?"
# → contains "it" → inject last context (NVIDIA)
# → LLM now knows it = NVIDIA ✅
    
def refiner_node(state: AgentState) -> AgentState:
    """Node 2 - Refine the query"""
    try:
        refined =refine_query(state["user_query"], state["intent"])
        return {"refined": refined}
    except Exception as e:
        return {"error": f"Query refinement failed: {e}"}
    
def fetcher_node(state: AgentState) -> AgentState:
    """Node 3 - Fetch real data"""
    try:
        fetched = fetch_data(
            state["user_query"],
            state["refined"],
            state["intent"]
        )
        # convert to dict for other storage
        return {"fetched": fetched if isinstance(fetched, dict) else fetched}
    except Exception as e:
        return {"error": f"Data fetching failed: {e}"}
    
def analysis_node(state: AgentState) -> AgentState:
    """Node 4 — Analyze fetched data"""
    try:
        fetched = state.get("fetched", {})
        
        # If no fetched data, create minimal structure for LLM to work with
        if not fetched or not fetched.get("results"):
            fetched = {
                "query": state["user_query"],
                "source_used": "llm",
                "results_count": 1,
                "results": [{
                    "title": "Context Query",
                    "content": f"Answer based on memory context: {state.get('memory_context', '')}. Query: {state['user_query']}",
                    "url": "internal",
                    "score": 1.0
                }]
            }
        
        analysis = analyze_data(fetched, state["intent"])
        return {"analysis": analysis.model_dump()}
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}
    
def response_node(state: AgentState) -> AgentState:
    """Node 5 — Generate final response"""
    try:
        # Guard against empty analysis
        if not state.get("analysis"):
            return {"response": "⚠️ AgentMind could not find enough information. Please try rephrasing your query."}
        
        from agents.models import AnalysisResult
        analysis_obj = AnalysisResult(**state["analysis"])
        response = generate_response(
            state["user_query"],
            analysis_obj,
            state["intent"]
        )
        return {"response": response}
    except Exception as e:
        return {"response": f"⚠️ AgentMind encountered an issue: {str(e)}. Please try again."}
# **What nodes are:**
# ```
# Each node is just a WRAPPER around your existing agent.
# It takes state → calls agent → returns updated state.

# return {"intent": intent}
# → doesn't replace entire state
# → only UPDATES the "intent" field
# → everything else stays as is


## Routing Logic

def should_fetch(state: AgentState) -> str:
    """
    Conditional edge — decides if we need to fetch data or not
    """
    if state.get("error"):
        return "response"
    
    intent = state.get("intent", {})
    memory = state.get("memory_context", "")
    
    # If memory context exists, always fetch for better answers
    if memory:
        return "fetcher"
    
    if intent.get("requires_search") or intent.get("requires_stock_data"):
        return "fetcher"
    
    return "analysis"


def check_error(state: AgentState) -> str:
    """
    After each node, check if error occurred
    """
    if state.get("error"):
        return "response"
    return "continue"

# **This is the INTELLIGENCE of LangGraph:**
# ```
# should_fetch() decides at runtime:
# → "explain quantum computing" 
#    requires_search = False → skip fetcher
#    saves API calls, faster response

# → "nvidia stock today"
#    requires_search = True → go to fetcher
#    gets real data

# This routing didn't exist in our old main.py

### Build The Graph
def build_graph():
    """
    Assembles all nodes and edges into the final workflow graph
    """

    ## initialize graph with our state
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("memory", memory_node)
    graph.add_node("intent", intent_node)
    graph.add_node("refiner", refiner_node)
    graph.add_node("fetcher", fetcher_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("response", response_node)

    ## memory is now entry point
    graph.set_entry_point("memory")            # CHANGE THIS
    graph.add_edge("memory", "intent") 

    ## Add fixed edges (Always happen)
    graph.add_edge("intent", "refiner")

    ## Add conditional edges (smart routing)
    graph.add_conditional_edges(
        "refiner",
        should_fetch,
        {
            "fetcher": "fetcher",
            "analysis": "analysis"
        }
    )

    ## After fetcher always goes to analysis
    graph.add_edge("fetcher", "analysis")

    # After analysis always goes to response
    graph.add_edge("analysis", "response")

    ## Response is the end
    graph.add_edge("response", END)

    return graph.compile()

# **Visual of what we just built:**
# ```
# [intent] → [refiner] → needs search? 
#                            YES → [fetcher] → [analysis] → [response] → END
#                            NO  →             [analysis] → [response] → END


## Run function + test
def run_graph(user_query: str) -> str:
    """
    Runs the compiled graph with a user query
    """
    graph = build_graph()
    
    initial_state = {
        "user_query": user_query,
        "intent": {},
        "refined": {},
        "fetched": {},
        "analysis": {},
        "response": "",
        "error": ""
    }
    
    result = graph.invoke(initial_state)
    return result.get("response", "Sorry, AgentMind could not process your query.")


if __name__ == "__main__":
    test_queries = [
        "whats happening with nvidia today",
        "explain how attention works in transformers"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"USER: {query}")
        print('='*60)
        response = run_graph(query)
        print(response)

# ## if want to see the graph in mermaid.live
# graph = build_graph()
# print(graph.get_graph().draw_mermaid())