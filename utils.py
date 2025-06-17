from transformers import pipeline
import requests
import os

from dotenv import load_dotenv
import streamlit as st


load_dotenv()

# FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_API_KEY =  st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY")) 
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Load once globally (or in startup if using FastAPI)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(articles):
    if not articles:
        return "No news available"

    sentiments = [sentiment_pipeline(article["title"])[0]["label"] for article in articles]

    if sentiments.count("POSITIVE") > sentiments.count("NEGATIVE"):
        return "Positive"
    elif sentiments.count("NEGATIVE") > sentiments.count("POSITIVE"):
        return "Negative"
    else:
        return "Mixed"



def fetch_financial_ratios(symbol: str) -> dict:
    try:
        url = f"{FMP_BASE_URL}/ratios-ttm/{symbol}?apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
       
       

        if not data:
            return {}

        ratios = data[0]
        return {
            "PE Ratio": ratios.get("peRatioTTM"),
            "ROE": ratios.get("returnOnEquityTTM"),
            "ROA": ratios.get("returnOnAssetsTTM"),
            "Current Ratio": ratios.get("currentRatioTTM"),
            "Debt/Equity": ratios.get("debtEquityRatioTTM")
        }
    except Exception as e:
        print(f"[fetch_financial_ratios] Failed for {symbol}: {e}")
        return {}


def fetch_analyst_ratings(symbol: str) -> str:
    try:
        url = f"{FMP_BASE_URL}/rating/{symbol}?apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            return "Not Available"

        rating = data[0]
        return (
            f"Rating: {rating.get('rating')}, "
            f"Score: {rating.get('ratingScore')}, "
            f"Recommendation: {rating.get('ratingRecommendation')}"
        )
    except Exception as e:
        print(f"[fetch_analyst_ratings] Failed for {symbol}: {e}")
        return "Not Available"

