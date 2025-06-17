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

def analyze_sentiment(news_headlines: list[str]) -> str:
    if not news_headlines:
        return "Neutral"

    # Sanitize input: remove None, empty strings, or non-str values
    clean_headlines = [str(h).strip() for h in news_headlines if isinstance(h, str) and h.strip()]
    
    if not clean_headlines:
        return "Neutral"

    try:
        results = sentiment_pipeline(clean_headlines)
        scores = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

        for res in results:
            label = res["label"]
            if label in scores:
                scores[label] += 1
            else:
                scores["NEUTRAL"] += 1  # fallback

        if scores["POSITIVE"] > scores["NEGATIVE"]:
            return "Positive"
        elif scores["NEGATIVE"] > scores["POSITIVE"]:
            return "Negative"
        return "Neutral"
    except Exception as e:
        print(f"[analyze_sentiment] Error analyzing sentiment: {e}")
        return "Neutral"







def fetch_financial_ratios(symbol: str) -> dict:
    try:
        url = f"{FMP_BASE_URL}/ratios-ttm/{symbol}?apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(data)
       

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

