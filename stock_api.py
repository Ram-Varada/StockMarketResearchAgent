import requests
import os
from dotenv import load_dotenv

load_dotenv()

STOCK_API_KEY = os.getenv("STOCK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_stock_data(symbol):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
    response = requests.get(url)
    data = response.json()

    try:
        quote = data["Global Quote"]
        return {
            "symbol": quote["01. symbol"],
            "open": quote["02. open"],
            "high": quote["03. high"],
            "low": quote["04. low"],
            "price": quote["05. price"],
            "volume": quote["06. volume"],
            "latest_trading_day": quote["07. latest trading day"],
            "previous_close": quote["08. previous close"],
            "change": quote["09. change"],
            "change_percent": quote["10. change percent"]
        }
    except KeyError:
        return None

def get_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data and data.get("articles"):
        return data["articles"][:5]  # Return top 5 news articles
    else:
        return []
