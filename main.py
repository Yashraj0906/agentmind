import sys
import os
from dotenv import load_dotenv

from agents.intent_agent import detect_intent
from agents.query_refiner import refine_query
from agents.data_fetcher import fetch_data
from agents.analysis_agent import analyze_data
from agents.response_agent import generate_response

load_dotenv()


# ##You type: "whats happening with nvidia"
# main.py handles everything:
# → calls intent agent
# → calls query refiner  
# → calls data fetcher
# → calls analysis agent
# → calls response agent
# → prints final answer

# Remove old imports of individual agents
# Add this single import instead
from evals.langsmith_tracer import run_with_tracing, metrics
from memory.memory_store import save_conversation

def run_agentmind(user_query: str) -> str:
    print(f"\n🤖 AgentMind Processing via LangGraph...")
    print(f"{'─'*50}")
    
    response = run_with_tracing(user_query)
    
    print(f"{'─'*50}")
    print("✅ Done!\n")
    
    # Save to memory
    try:
        save_conversation(user_query, response)
    except:
        pass
    
    return response

## Conversation Loop

def main():
    """
    Interactive conversation loop
    Keeps running until user types 'exit'
    """
    
    print("\n" + "="*60)
    print("🧠  AGENTMIND — Universal Intelligence System")
    print("="*60)
    print("Ask me anything. Type 'exit' to quit.\n")
    
    # Conversation history for context
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAgentMind: Goodbye! ")
            metrics.print_summary()
            break
        
        # Skip empty input
        if not user_input:
            continue
        
        try:
            # Run the full pipeline
            response = run_agentmind(user_input)
            
            # Store in conversation history
            conversation_history.append({
                "user": user_input,
                "agent": response
            })
            
            # Print response
            print(f"\n{response}\n")
            
        except Exception as e:
            print(f"\n AgentMind encountered an error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()


# **Key concept — Conversation Loop:**
# ```
# while True loop keeps running forever
# → user types query
# → pipeline runs
# → response printed
# → loop repeats

# This is what makes it a CONVERSATIONAL system
# not just a one-shot script

# try/except around entire pipeline
# → if anything fails, system doesn't crash
# → tells user what went wrong
# → ready for next query immediately