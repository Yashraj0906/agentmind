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


# core pipeline function
def run_agentmind(user_query: str) -> str:
    """
    Master pipeline - runs all agents in sequence
    Takes user query, returns final response
    """

    print(f"\n🤖 AgentMind Processing...")
    print(f"{'-'*50}")

    # step 1 - understand intent
    print(" Step 1/5: Detecting Intent...")
    intent = detect_intent(user_query)
    print(f"   Type: {intent['query_type']} | Domain: {intent['domain']}")

    #step 2 - refine query
    print("Step 2/5: Refining query...")
    refined = refine_query(user_query, intent)
    print(f"   Search: {refined['primary_query']}")

    # step 3 - Fetch Data
    print("Step 3/5: Fetching Data...")
    fetched = fetch_data(user_query, refined, intent)
    print(f"   Source: {fetched['source_used']} | Results: {fetched['results_count']}")

    # step 4 - analyze
    print("Step 4/5: Analyzing results...")
    analysis = analyze_data(fetched, intent)
    print(f"   Sentiment: {analysis.sentiment} | Confidence: {analysis.confidence}")

    # Step 5 — Generate response
    print("Step 5/5: Generating response...")
    response = generate_response(user_query, analysis, intent)

    print(f"{'─'*50}")
    print("Done!\n")

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