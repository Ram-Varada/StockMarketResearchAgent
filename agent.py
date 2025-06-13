from llm import get_llm
from stock_api import fetch_stock_data, get_news
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.language_models import BaseLLM, BaseLanguageModel
from langchain.chains import LLMChain
from typing import List, Dict, Any
import re
import json


def stock_research_agent(query):
    llm = get_llm()

    # For simplicity, extract company symbol from query here
    # You can improve with NLP parsing later
    symbol = None
    if "apple" in query.lower():
        symbol = "AAPL"
    elif "tesla" in query.lower():
        symbol = "TSLA"
    else:
        return "Sorry, I currently support Apple and Tesla only."

    stock_info = fetch_stock_data(symbol)
    news = get_news(symbol)

    if not stock_info:
        return f"Sorry, no data found for {symbol}"

    # Create a prompt with stock info + news
    news_summaries = "\n".join([article["title"] for article in news])
    
    
    prompt = f"""
    You are a stock market research assistant.

    Stock info for {symbol}:
    - Price: {stock_info['price']}
    - Change: {stock_info['change']} ({stock_info['change_percent']})
    - Open: {stock_info['open']}
    - High: {stock_info['high']}
    - Low: {stock_info['low']}
    - Volume: {stock_info['volume']}

    Recent news:
    {news_summaries}

    Provide an insightful summary and investment outlook for this company based on the data and news.
    """

    
    
    answer = llm.invoke(prompt)  # ✅ correct method

    return answer

def generate_summary(stock_data, news_data):
    news_text = "\n".join([n["title"] for n in news_data]) if news_data else "No news available."
    prompt = PromptTemplate.from_template("""
    You are a stock research analyst.

    Stock Info:
    {stock_data}

    News:
    {news_text}

    Give an investment summary and outlook.
    """)
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"stock_data": stock_data, "news_text": news_text})






def extract_company_name(llm: BaseLLM, query: str) -> str:
    prompt = PromptTemplate.from_template(
        "Extract the company or stock name from the user's query:\n\n"
        "Query: {query}\n"
        "Answer:"
    )

    chain = prompt | llm  # prompt and llm are Runnable
    result = chain.invoke({"query": query})
    print(result)
    
    # Make sure we print the full object to inspect in debug
    print("LLM raw result:", result)

    # If result is an AIMessage, get content
    if hasattr(result, "content"):
        result = result.content

    return result.strip().replace('"', '')

def extract_intent_and_symbols(llm: BaseLanguageModel, query: str) -> Dict[str, Any]:
    prompt = f"""
    You are an intelligent assistant helping categorize user queries related to stock market analysis.

    Given the following query, identify:
    1. The user's intent - one of: "stock_data", "news", "both", or "compare_stocks".
    2. The official stock **ticker symbols** (like "AAPL", "MSFT") mentioned in the query.

    Return your answer as a JSON object with:
    - "intent": one of the valid intent values.
    - "symbols": a list of **ticker symbols** (1 or 2 max).

    Respond ONLY with the JSON block. No extra explanation.

    Query: "{query}"
    """


    response = llm.invoke(prompt)
    content = response.content

    print('llm raw response content:', content)

    # Extract JSON block inside ```json ... ```
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not json_match:
        # fallback: try to parse the whole content anyway (in case no backticks)
        json_str = content.strip()
    else:
        json_str = json_match.group(1)

    try:
        result = json.loads(json_str)
        print("[extract_intent_and_symbols] Parsed JSON:", result)
        intent = result.get("intent", "both")
        symbols = result.get("symbols", [])
        return {"intent": intent, "symbols": symbols}
    except Exception as e:
        print("[extract_intent_and_symbols] Failed to parse JSON. Using fallback.", e)
        return {"intent": "both", "symbols": [query.strip()]}


def compare_stock_summaries(symbols: List[str]) -> str:
    from stock_api import fetch_stock_data
    if len(symbols) != 2:
        return "Comparison requires exactly 2 symbols."

    data_1 = fetch_stock_data(symbols[0])
    data_2 = fetch_stock_data(symbols[1])

    def format_data(symbol: str, data: Dict[str, Any]) -> str:
        return f"**{symbol.upper()}**\n- Price: ${data.get('price')}\n- High: ${data.get('high')}\n- Low: ${data.get('low')}\n- Volume: {data.get('volume')}"

    return f"""
    ## 📊 Stock Comparison: {symbols[0].upper()} vs {symbols[1].upper()}

    {format_data(symbols[0], data_1)}

    ---

    {format_data(symbols[1], data_2)}

    📌 **Note:** This is a raw financial comparison. Consider reading latest news and analyst insights for better investment decisions.
    """
    

