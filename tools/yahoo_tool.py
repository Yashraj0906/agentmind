import yfinance as yf


def get_stock_data(ticker: str) -> dict:
    """
    Gets real stock data for any ticker — US or Indian
    US:     NVDA, AAPL, TSLA, MSFT
    Indian: RELIANCE.NS, TCS.NS, INFY.NS
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "success": True
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_ticker_from_query(query: str) -> str:
    """
    Maps company names to ticker symbols
    US and Indian companies
    """
    mapping = {
        # US Stocks
        "nvidia": "NVDA",
        "apple": "AAPL",
        "google": "GOOGL",
        "microsoft": "MSFT",
        "tesla": "TSLA",
        "amazon": "AMZN",
        "meta": "META",
        "netflix": "NFLX",
        "amd": "AMD",
        "intel": "INTC",
        
        # Indian Stocks
        "reliance": "RELIANCE.NS",
        "tcs": "TCS.NS",
        "infosys": "INFY.NS",
        "wipro": "WIPRO.NS",
        "hdfc": "HDFCBANK.NS",
        "icici": "ICICIBANK.NS",
        "sbi": "SBIN.NS",
        "adani": "ADANIENT.NS",
        "tata motors": "TATAMOTORS.NS",
        "bajaj": "BAJFINANCE.NS"
    }
    
    query_lower = query.lower()
    for company, ticker in mapping.items():
        if company in query_lower:
            return ticker
    return None