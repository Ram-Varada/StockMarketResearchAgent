from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Annotated
from langchain_core.runnables import Runnable
from agent import generate_summary, extract_company_name, extract_intent_and_symbols,compare_stock_summaries
from stock_api import fetch_stock_data, get_news
from llm import get_llm
from utils import analyze_sentiment,fetch_financial_ratios,fetch_analyst_ratings

# ---------------------------
# Define the agent's state
# ---------------------------
class AgentState(TypedDict):
    user_query: Annotated[str, "single"]
    symbol: Annotated[str, "single"]
    user_intent: Annotated[Literal["stock_data", "news", "both", "compare_stocks"], "single"]
    stock_data: Annotated[str, "single"]
    news_data: Annotated[str, "single"]
    summary: Annotated[str, "single"]
    compare_symbols: Annotated[list[str], "single"]
    comparison_result: Annotated[str, "single"]

# ---------------------------
# Node: Analyze user query
# ---------------------------
def analyze_user_query_node(state: AgentState) -> AgentState:
    query = state["user_query"].lower()
    llm = get_llm()

    result = extract_intent_and_symbols(llm, query)
    print('result',result)
    intent = result["intent"]
    symbols = result["symbols"]

    print(f"[Analyze] Intent: {intent}, Symbols: {symbols}")

    if intent == "compare_stocks":
        return {**state, "user_intent": intent, "compare_symbols": symbols}
    else:
        return {**state, "user_intent": intent, "symbol": symbols[0]}

# ---------------------------
# Node: Fetch stock data
# ---------------------------
def fetch_data_node(state: AgentState) -> AgentState:
    symbol = state.get("symbol")
    data = fetch_stock_data(symbol)
    print(f"[Fetch Data] Fetched for {symbol}: {str(data)[:100]}...")
    return {**state, "stock_data": data}

# ---------------------------
# Node: Fetch news
# ---------------------------
def fetch_news_node(state: AgentState) -> AgentState:
    symbol = state.get("symbol")
    news = get_news(symbol)
    print(f"[Fetch News] Fetched for {symbol}: {len(news)} articles")
    return {**state, "news_data": news}

# ---------------------------
# Node: Summarize
# ---------------------------
def summarize_node(state: AgentState) -> AgentState:
    print(f"[Summarize] Intent: {state['user_intent']}")
    news = state.get("news_data", []) if state["user_intent"] != "stock_data" else []
    stock_data = state.get("stock_data", "")
    summary = generate_summary(stock_data, news)
    print("===== Summary =====\n")
    print(summary)
    return {**state, "summary": summary}

# ---------------------------
# Node: Compare Stocks
# ---------------------------



def compare_stocks_node(state: AgentState) -> AgentState:
    llm = get_llm()
    symbols = state.get("compare_symbols", [])
    print("symbols", symbols)

    if len(symbols) < 2:
        return {**state, "summary": "Please provide two stock symbols to compare."}

    symbol_a, symbol_b = symbols[0], symbols[1]

    # Fetch data for both stocks
    stock_data_a = fetch_stock_data(symbol_a)
    stock_data_b = fetch_stock_data(symbol_b)

    news_a = get_news(symbol_a)
    news_b = get_news(symbol_b)

    # Optional: Sentiment, Ratios, Analyst Ratings
    sentiment_a = analyze_sentiment(news_a) if callable(globals().get("analyze_sentiment")) else "Neutral"
    sentiment_b = analyze_sentiment(news_b) if callable(globals().get("analyze_sentiment")) else "Neutral"

    ratios_a = fetch_financial_ratios(symbol_a) if callable(globals().get("fetch_financial_ratios")) else {}
    ratios_b = fetch_financial_ratios(symbol_b) if callable(globals().get("fetch_financial_ratios")) else {}

    analyst_ratings_a = fetch_analyst_ratings(symbol_a) if callable(globals().get("fetch_analyst_ratings")) else "Not Available"
    analyst_ratings_b = fetch_analyst_ratings(symbol_b) if callable(globals().get("fetch_analyst_ratings")) else "Not Available"

    # Format Financial Ratio Table
    financial_metrics = ["P/E Ratio", "Dividend Yield", "Market Cap", "Revenue Growth YoY", "Net Margin"]
    ratio_table = "| Metric | " + symbol_a + " | " + symbol_b + " |\n|--------|" + "--------|"*2 + "\n"
    for metric in financial_metrics:
        val_a = ratios_a.get(metric, "N/A")
        val_b = ratios_b.get(metric, "N/A")
        ratio_table += f"| {metric} | {val_a} | {val_b} |\n"

    # Create structured LLM prompt
    prompt = f"""
    You are a stock market analyst comparing two companies based on their metrics, news, and financial sentiment.

    Compare {symbol_a} and {symbol_b} across these sections:

    1. 📊 **Financial Overview Table**  
    {ratio_table}

    2. 📌 **Company Strengths & Weaknesses**  
    Use your knowledge and recent news headlines to list 2-3 strengths and weaknesses for each company.

    3. 🔍 **Recent Trends & Outlook**  
    Summarize key business or market movements, earnings news, product launches, and growth plans.

    4. 📉 **News Sentiment Score (Past 7 Days)**  
    Apple: {sentiment_a}  
    Microsoft: {sentiment_b}

    5. 🧠 **Analyst Rating Summary**  
    {symbol_a}: {analyst_ratings_a}  
    {symbol_b}: {analyst_ratings_b}

    6. 👥 **Who Should Invest in Each?**  
    For {symbol_a}: Describe ideal investor profiles.  
    For {symbol_b}: Same.

    7. ✅ **Final Verdict**  
    Conclude with a recommendation: Which stock may be better and why, based on risk/reward.
    """.strip()

    analysis_summary = llm.invoke(prompt).content.strip()

    formatted_output = f"## 📊 Stock Comparison: {symbol_a} vs {symbol_b}\n\n{analysis_summary}"

    return {**state, "summary": formatted_output}


# ---------------------------
# Router
# ---------------------------
def analyze_router(state: AgentState) -> str:
    return state["user_intent"]

# ---------------------------
# Build the LangGraph
# ---------------------------
def build_stock_graph() -> Runnable:
    builder = StateGraph(AgentState)

    builder.add_node("analyze_query", analyze_user_query_node)
    builder.add_node("fetch_data", fetch_data_node)
    builder.add_node("fetch_news", fetch_news_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("compare_stocks", compare_stocks_node)

    builder.set_entry_point("analyze_query")

    builder.add_conditional_edges(
        "analyze_query",
        analyze_router,
        {
            "stock_data": "fetch_data",
            "news": "fetch_news",
            "both": "fetch_data",
            "compare_stocks": "compare_stocks"
        }
    )

    builder.add_conditional_edges(
        "fetch_data",
        lambda state: state["user_intent"],
        {
            "stock_data": "summarize",
            "both": "fetch_news"
        }
    )

    builder.add_edge("fetch_news", "summarize")
    builder.add_edge("summarize", END)
    builder.add_edge("compare_stocks", END)

    return builder.compile()