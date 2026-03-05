from pydantic import BaseModel, Field
from typing import List, Optional

class IntentResult(BaseModel):
    query_type: str
    domain: str
    requires_search: bool
    requires_stock_data: bool
    timeframe: str
    confidence: str

class RefinedQuery(BaseModel):
    primary_query: str
    secondary_query: str
    keywords: List[str]
    search_type: str

class SearchResult(BaseModel):
    title: str
    content: str
    url: str
    score: float

class FetchedData(BaseModel):
    query: str
    source_used: str
    results_count: int
    results: List[SearchResult]

class AnalysisResult(BaseModel):
    query: str
    summary: str
    key_insights: List[str]
    sentiment: str
    confidence: str
    sources: List[str]
    follow_up_suggestions: List[str]

# **What this does:**
# ```
# BaseModel        → Pydantic's base class
# Field(...)       → adds validation rules
# List[str]        → must be a list of strings
# Optional[str]    → can be None

# Now every agent knows EXACTLY what shape
# of data to expect and produce.
# No more silent None bugs.