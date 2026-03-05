import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"


def get_stock_price(ticker: str) -> dict:
    """
    Gets latest stock data for a ticker symbol
    Example: get_stock_price("NVDA")
    """
    try:
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        params = {"apiKey": API_KEY}
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("results"):
            result = data["results"][0]
            return {
                "ticker": ticker,
                "close_price": result.get("c"),
                "open_price": result.get("o"),
                "high": result.get("h"),
                "low": result.get("l"),
                "volume": result.get("v"),
                "success": True
            }
        return {"success": False, "error": "No data found"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_ticker_from_query(query: str) -> str:
    """
    Common company to ticker mapping
    """
    mapping = {
        "nvidia": "NVDA",
        "apple": "AAPL",
        "google": "GOOGL",
        "microsoft": "MSFT",
        "tesla": "TSLA",
        "amazon": "AMZN",
        "meta": "META",
        "openai": None  # Not public
    }
    
    query_lower = query.lower()
    for company, ticker in mapping.items():
        if company in query_lower:
            return ticker
    return None