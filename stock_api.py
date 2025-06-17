import requests
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

""" STOCK_API_KEY = os.getenv("STOCK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY") """

STOCK_API_KEY = st.secrets.get("STOCK_API_KEY", os.getenv("STOCK_API_KEY"))
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY")) 
FMP_API_KEY =  st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY"))  

import requests

def fetch_stock_data(symbol: str) -> dict:    
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        

        if not data:
            print(f"[FMP] No data returned for symbol: {symbol}")
            return None

        quote = data[0]

        return {
            "symbol": quote.get("symbol"),
            "price": quote.get("price"),
            "open": quote.get("open"),
            "high": quote.get("dayHigh"),
            "low": quote.get("dayLow"),
            "volume": quote.get("volume"),
            "previous_close": quote.get("previousClose"),
            "change": quote.get("change"),
            "change_percent": f"{quote.get('changesPercentage', 0):.2f}%",
            "latest_trading_day": quote.get("timestamp")  # Could convert this to readable format
        }

    except Exception as e:
        print(f"[FMP] Error fetching stock data for {symbol}: {e}")
        return None


def get_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data and data.get("articles"):
        return data["articles"][:5]  # Return top 5 news articles
    else:
        return []
