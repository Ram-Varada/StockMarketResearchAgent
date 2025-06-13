from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Annotated
from langchain_core.runnables import Runnable
from agent import generate_summary, extract_company_name, extract_intent_and_symbols,compare_stock_summaries
from stock_api import fetch_stock_data, get_news
from llm import get_llm

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
    print('symbols', symbols)
    if len(symbols) < 2:
        # fallback if only one or no symbols recognized
        return {**state, "summary": "Please provide two stock symbols to compare."}

    # Fetch raw data for both stocks
    stock_a_data = fetch_stock_data(symbols[0])
    stock_b_data = fetch_stock_data(symbols[1])

    # Optional: fetch news summaries for context
    news_a = get_news(symbols[0])
    news_b = get_news(symbols[1])

    # Prepare a prompt for LLM to generate a meaningful comparison
    prompt = f"""
    You are a financial analyst. Compare these two stocks based on the data and recent news headlines.

    Stock A ({symbols[0]}):
    {stock_a_data}

    Recent news headlines:
    {news_a[:5]}

    Stock B ({symbols[1]}):
    {stock_b_data}

    Recent news headlines:
    {news_b[:5]}

    Provide a detailed comparative analysis highlighting strengths, weaknesses, recent trends, volatility, and overall outlook. Conclude with which stock might be a better investment and why.
    """

    # Ask LLM for the analysis summary
    analysis_summary = llm.invoke(prompt).content.strip()

    # Format final summary with heading
    final_summary = f"## 📊 Stock Comparison: {symbols[0]} vs {symbols[1]}\n\n{analysis_summary}"

    return {**state, "summary": final_summary}


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