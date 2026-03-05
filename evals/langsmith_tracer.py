import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langsmith import Client
from langsmith import traceable

load_dotenv()

client = Client()

# **What this does:**
# ```
# Client()     → connects to your LangSmith account
#               using LANGSMITH_API_KEY from .env
# traceable    → decorator that auto-traces any function
#               you put @traceable above


## Metrics Tracker
class AgentMetrics:
    """
    Tracks performance metrics across all agent runs
    These numbers become your resume bullet points
    """
    
    def __init__(self):
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self.source_usage = {
            "tavily": 0,
            "tavily_fallback": 0,
            "llm": 0,
            "llm_fallback": 0
        }
        self.response_times = []
        self.query_types = {}
    
    def record_run(self, success: bool, source: str, 
                   response_time: float, query_type: str):
        self.total_runs += 1
        
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        
        if source in self.source_usage:
            self.source_usage[source] += 1
            
        self.response_times.append(response_time)
        
        self.query_types[query_type] = self.query_types.get(query_type, 0) + 1
    
    def get_summary(self) -> dict:
        if self.total_runs == 0:
            return {"message": "No runs recorded yet"}
        
        avg_time = sum(self.response_times) / len(self.response_times)
        success_rate = (self.successful_runs / self.total_runs) * 100
        
        return {
            "total_runs": self.total_runs,
            "success_rate": f"{success_rate:.1f}%",
            "avg_response_time": f"{avg_time:.2f}s",
            "source_breakdown": self.source_usage,
            "query_types": self.query_types,
            "failed_runs": self.failed_runs
        }
    
    def print_summary(self):
        summary = self.get_summary()
        print("\n" + "="*50)
        print("📊 AGENTMIND PERFORMANCE METRICS")
        print("="*50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("="*50)

# Global metrics instance
metrics = AgentMetrics()

## Traceable Pipeline
@traceable(name="AgentMind-Full-Pipeline")
def traced_pipeline(user_query: str) -> dict:
    """
    Full pipeline wrapped with LangSmith tracing.
    Every run is automatically logged to LangSmith dashboard.
    """
    from graph.workflow import run_graph
    
    start_time = time.time()
    success = True
    source_used = "unknown"
    query_type = "unknown"
    
    try:
        # Run the full LangGraph pipeline
        response = run_graph(user_query)
        
        # Check if response is valid
        if not response or "could not find" in response.lower():
            success = False
            
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "query": user_query,
            "response": response,
            "success": success,
            "response_time": response_time
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            "query": user_query,
            "response": f"Error: {str(e)}",
            "success": False,
            "response_time": end_time - start_time
        }


def run_with_tracing(user_query: str) -> str:
    """
    Main function to call — runs pipeline with full tracing and metrics
    """
    start = time.time()
    
    result = traced_pipeline(user_query)
    
    # Record metrics
    metrics.record_run(
        success=result["success"],
        source="tavily",
        response_time=result["response_time"],
        query_type="general"
    )
    
    return result["response"]

## Test + Eval Runner

if __name__ == "__main__":
    
    print("🔍 Running AgentMind Eval Suite...")
    print("All runs will appear in LangSmith dashboard\n")
    
    test_queries = [
        "What is happening in AI industry today?",
        "Explain what is machine learning",
        "Latest news on OpenAI",
        "Compare Python vs JavaScript",
        "What is NVIDIA stock doing?"
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        result = run_with_tracing(query)
        status = "✅" if result and "Error" not in result else "❌"
        print(f"Status: {status}")
        print("-" * 40)
    
    # Print final metrics
    metrics.print_summary()