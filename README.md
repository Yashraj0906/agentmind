# 🧠 AgentMind — Universal Multi-Agent Intelligence System

> Ask anything. AgentMind figures out what you need, finds the right data, thinks about it, and gives you a structured, intelligent answer.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3--70B-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 What is AgentMind?

AgentMind is a **universal multi-agent AI system** that coordinates 6 specialized agents to answer any query intelligently — from real-time stock prices and market news to technical explanations and general knowledge.

It is **not a chatbot**. It is **not a search engine**.
It is a team of AI agents working together — the same architectural pattern used by Bloomberg, Morgan Stanley, and leading AI-first companies in 2026.

---

## 🎬 Demo

```
You: What is happening with NVIDIA stock today?

🤖 AgentMind Processing via LangGraph...
──────────────────────────────────────────
## 🔍 NVIDIA Stock Update

**Summary**
NVIDIA's stock is trading around $183, driven by continued AI 
infrastructure demand. GTC 2026 conference expected as major catalyst.

**Key Insights**
• Stock trading at $183.04 with day range $180.06 - $184.70
• $41.1B returned to shareholders, $58.5B remaining in buyback
• GTC 2026 (March 16-19) — Jensen Huang keynote expected

**Sentiment:** Neutral | **Confidence:** High

**You might also want to ask:**
→ What announcements are expected at NVIDIA GTC 2026?
→ How does NVIDIA compare to AMD in AI chip market share?

---
*Powered by AgentMind*
```

---

## 🏗️ Architecture

AgentMind uses a **LangGraph stateful workflow** with conditional routing — agents are nodes, connections are edges, and a shared `AgentState` acts as the whiteboard all agents read from and write to.

```
User Query
    ↓
[Memory Node]        → Fetches relevant context from past conversations
    ↓
[Intent Agent]       → Classifies query type, domain, required tools
    ↓
[Query Refiner]      → Turns vague input into precise searchable query
    ↓
    ├── requires_search = True  → [Data Fetcher] → Tavily + Yahoo Finance
    └── requires_search = False → skip fetcher (saves API calls)
    ↓
[Analysis Agent]     → Extracts key insights, sentiment, confidence
    ↓
[Response Agent]     → Formats structured, readable final answer
```

### LangGraph Flow Diagram


<img width="1919" height="1029" alt="Screenshot 2026-03-06 012105" src="https://github.com/user-attachments/assets/eb2f454a-b63c-4eac-9514-589cab206084" />

> *The dotted lines show conditional routing — the system decides at runtime whether to fetch external data or answer directly from LLM knowledge.*

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq (LLaMA 3.3 70B) | Fast inference for all agents |
| **Orchestration** | LangGraph | Stateful multi-agent workflow |
| **Web Search** | Tavily | Real-time news and web data |
| **Stock Data (US)** | Polygon.io | US market financial data |
| **Stock Data (US+India)** | Yahoo Finance (yfinance) | NSE/BSE + US stocks |
| **Data Validation** | Pydantic | Type-safe agent communication |
| **Memory** | In-memory store | Multi-turn conversation context |
| **Observability** | LangSmith | Agent tracing and eval metrics |
| **UI** | Streamlit | Interactive chat interface |
| **Environment** | uv | Fast Python package management |

---

## 📊 Performance Metrics

| Metric | Value |
|---|---|
| Response Success Rate | **100%** (5-query eval suite) |
| Average Response Time | **4.72 seconds** |
| Agents Coordinated | **6** |
| Fallback Layers | **3** (primary → secondary → LLM) |
| Supported Domains | Finance, Tech, News, General, Comparison |
| Stock Markets | US (NYSE/NASDAQ) + India (NSE/BSE) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- uv package manager
- API keys (see below)

### Installation

```bash
# Clone the repo
git clone https://github.com/Yashraj0906/agentmind.git
cd agentmind

# Install dependencies
uv add langchain langgraph langchain-groq tavily-python \
       mem0ai langsmith python-dotenv streamlit wikipedia \
       pydantic yfinance

# Setup environment variables
cp .env.example .env
# Add your API keys to .env
```

### API Keys Required

| Service | Get Key At | Free Tier |
|---|---|---|
| Groq | console.groq.com | ✅ Yes |
| Tavily | tavily.com | ✅ 1000 searches/month |
| Polygon.io | polygon.io | ✅ 5 calls/min |
| LangSmith | smith.langchain.com | ✅ 5k traces/month |

### Environment Variables

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
POLYGON_API_KEY=your_polygon_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agentmind
```

### Run

```bash
# Terminal mode
uv run python main.py

# Web UI
uv run streamlit run app.py

# Run eval suite
uv run python evals/langsmith_tracer.py
```

---

## 📁 Project Structure

```
agentmind/
├── agents/
│   ├── intent_agent.py       # Classifies query type and domain
│   ├── query_refiner.py      # Optimizes queries for search
│   ├── data_fetcher.py       # Tavily + Yahoo Finance + fallbacks
│   ├── analysis_agent.py     # Extracts insights from raw data
│   ├── response_agent.py     # Formats final response
│   └── models.py             # Pydantic data models
├── graph/
│   └── workflow.py           # LangGraph stateful workflow
├── memory/
│   └── memory_store.py       # Multi-turn conversation memory
├── tools/
│   ├── yahoo_tool.py         # Yahoo Finance integration
│   └── polygon_tool.py       # Polygon.io integration
├── evals/
│   └── langsmith_tracer.py   # LangSmith tracing + metrics
├── app.py                    # Streamlit chat UI
├── main.py                   # Terminal entry point
└── .env                      # API keys (never committed)
```

---

## 🔍 Supported Query Types

| Query Type | Example | Data Source |
|---|---|---|
| **Market** | "What is NVIDIA stock today?" | Yahoo Finance + Tavily |
| **Indian Stocks** | "What is Reliance stock price?" | Yahoo Finance (NSE) |
| **News** | "Latest news on OpenAI" | Tavily web search |
| **Technology** | "Explain transformer attention" | LLM knowledge |
| **Comparison** | "Compare Python vs JavaScript" | LLM knowledge |
| **General** | "Who is PM of India?" | Tavily web search |

---

## 🧪 Running Evals

```bash
uv run python evals/langsmith_tracer.py
```

Output:
```
📊 AGENTMIND PERFORMANCE METRICS
==================================================
total_runs: 5
success_rate: 100.0%
avg_response_time: 4.72s
source_breakdown: {'tavily': 5, 'llm': 0}
failed_runs: 0
```

View detailed traces at: **smith.langchain.com** → agentmind project


## 👤 Author

**Yashraj Kumar**
3rd Year ECE Student | IIIT Nagpur
Interested in AI/ML, Agentic Systems, and Generative AI

[![GitHub](https://img.shields.io/badge/GitHub-Yashraj0906-black?style=flat-square&logo=github)](https://github.com/Yashraj0906)

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.

---

*Built with LangGraph + Groq + Tavily | March 2026*
