from llm import get_llm
from stock_api import fetch_stock_data, get_news
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.language_models import BaseLLM
from langchain.chains import LLMChain

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
    

