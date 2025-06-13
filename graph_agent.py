from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_core.runnables import Runnable
from agent import generate_summary, extract_company_name
from stock_api import fetch_stock_data, get_news
from llm import get_llm

from typing import Annotated

# ---------------------------
# Define the agent's state
# ---------------------------
class AgentState(TypedDict):
    user_query: Annotated[str, "single"]
    symbol: Annotated[str, "single"]
    user_intent: Annotated[Literal["stock_data", "news", "both"], "single"]
    stock_data: Annotated[str, "single"]
    news_data: Annotated[str, "single"]
    summary: Annotated[str, "single"]



# ---------------------------
# Node: Analyze user query
# ---------------------------
def analyze_user_query_node(state: AgentState) -> AgentState:
    query = state["user_query"].lower()
    llm = get_llm()

    extracted = extract_company_name(llm, query)

    if "news" in query:
        intent = "news"
    elif "price" in query or "stock" in query:
        intent = "stock_data"
    else:
        intent = "both"

    print(f"[Analyze] Extracted symbol: {extracted}, Intent: {intent}")
    return {**state, "symbol": extracted, "user_intent": intent}

# ---------------------------
# Node: Fetch stock data
# ---------------------------
def fetch_data_node(state: AgentState) -> AgentState:
    symbol = state.get("symbol")
    data = fetch_stock_data(symbol)
    
    print(f"[Fetch Data] Fetched for {symbol}: {str(data)[:100]}...")  # Preview
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
    print(f"[Summarize] News data exists? {'Yes' if state.get('news_data') else 'No'}")

    news = state.get("news_data", []) if state["user_intent"] != "stock_data" else []
    stock_data = state.get("stock_data", "")
    summary = generate_summary(stock_data, news)

    print("===== Summary =====\n")
    print(summary)

    return {**state, "summary": summary}

# ---------------------------
# Router: After analyze
# ---------------------------
def analyze_router(state: AgentState) -> str:
    return state["user_intent"]

# ---------------------------
# Build the LangGraph
# ---------------------------
def build_stock_graph() -> Runnable:
    builder = StateGraph(AgentState)

    # Add all nodes
    builder.add_node("analyze_query", analyze_user_query_node)
    builder.add_node("fetch_data", fetch_data_node)
    builder.add_node("fetch_news", fetch_news_node)
    builder.add_node("summarize", summarize_node)

    builder.set_entry_point("analyze_query")

    # Branch after analyze
    builder.add_conditional_edges(
        "analyze_query",
        analyze_router,
        {
            "stock_data": "fetch_data",
            "news": "fetch_news",
            "both": "fetch_data",
        }
    )

    # Branch after fetch_data based on intent
    builder.add_conditional_edges(
        "fetch_data",
        lambda state: state["user_intent"],
        {
            "stock_data": "summarize",  # only summarize directly if no news needed
            "both": "fetch_news"        # continue to news first if both
        }
    )

    builder.add_edge("fetch_news", "summarize")
    builder.add_edge("summarize", END)


    return builder.compile()
