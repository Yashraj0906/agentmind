import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from evals.langsmith_tracer import run_with_tracing, metrics
from memory.memory_store import save_conversation

load_dotenv()

#Page Config
st.set_page_config(
    page_title="AgentMind",
    page_icon="🧠",
    layout="centered"
)

# Header
st.title("🧠 AgentMind")
st.caption("Universal Intelligence System — powered by LangGraph + Groq")
st.divider()

#Session State Init
# Initialize chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# **What session_state is:**
# ```
# Streamlit reruns entire script on every interaction.
# session_state persists data across reruns.
# Without it, chat history would reset every message.

### Sidebar Metrics
with st.sidebar:
    st.header("📊 Session Metrics")
    st.metric("Queries Processed", st.session_state.total_queries)
    
    summary = metrics.get_summary()
    if st.session_state.total_queries > 0:
        st.metric("Success Rate", summary.get("success_rate", "N/A"))
        st.metric("Avg Response Time", summary.get("avg_response_time", "N/A"))
        
        st.subheader("Source Usage")
        breakdown = summary.get("source_breakdown", {})
        for source, count in breakdown.items():
            if count > 0:
                st.write(f"• {source}: {count}")
    
    st.divider()
    st.caption("Built with LangGraph + Groq + Tavily")
    st.caption("By Yashraj Kumar")

## Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#Chat Input + Processing
if prompt := st.chat_input("Ask AgentMind anything..."):
    
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and show response
    with st.chat_message("assistant"):
        with st.spinner("🤖 AgentMind thinking..."):
            try:
                response = run_with_tracing(prompt)
                
                # Save to memory
                save_conversation(prompt, response)
                
                # Update metrics
                st.session_state.total_queries += 1
                
                st.markdown(response)
                
                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })