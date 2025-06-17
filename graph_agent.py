from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Annotated
from langchain_core.runnables import Runnable
from agent import  extract_intent_and_symbols
from stock_api import fetch_stock_data, get_news
from llm import get_llm
from utils import analyze_sentiment,fetch_financial_ratios,fetch_analyst_ratings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
today = datetime.now().strftime("%B %d, %Y")



# ---------------------------
# Define the agent's state
# ---------------------------
class AgentState(TypedDict):
    user_query: Annotated[str, "single"]
    symbols: Annotated[str, "single"]
    user_intent: Annotated[Literal["gather_stock_info",  "compare_stocks"], "single"]
    stock_info: Annotated[str, "single"]
    news_info: Annotated[str, "single"]
    summary: Annotated[str, "single"]
    compare_symbols: Annotated[list[str], "single"]
    last_symbols: Annotated[list[str], "single"]
    comparison_result: Annotated[str, "single"]

# ---------------------------
# Node: Analyze user query
# ---------------------------
def analyze_user_query_node(state: AgentState) -> AgentState:
    query = state["user_query"].lower()
    llm = get_llm()

    result = extract_intent_and_symbols(llm, query)
    
    intent = result["intent"]
    symbols = result["symbols"]

    

    if intent == "compare_stocks":
        return {**state, "user_intent": intent, "compare_symbols": symbols}
    else:
        return {**state, "user_intent": intent, "symbols": symbols[0]}

   

# ---------------------------
# Node: Fetch news
# ---------------------------
def fetch_news_node(state: AgentState) -> AgentState:
    
    symbol = state.get("symbols", [])
    news = get_news(symbol)
   
    return {**state, "news_data": news}


def  gather_stock_info_node(state: AgentState) -> AgentState:
   
    symbol = state.get("symbols", [])
      
    
    
    stock_data = fetch_stock_data(symbol)
    financial_ratios = fetch_financial_ratios(symbol)
 
    news_info = (fetch_news_node(state))

    combined_data = {
        "Current Price": stock_data.get("price"),
        "52-Week Range": f"{stock_data.get('low')} - {stock_data.get('high')}",
        "Volume": stock_data.get("volume"),
        "P/E Ratio": financial_ratios.get("PE Ratio"),
        "ROE": financial_ratios.get("ROE"),
        "ROA": financial_ratios.get("ROA"),
        "Current Ratio": financial_ratios.get("Current Ratio"),
        "Debt-to-Equity": financial_ratios.get("Debt/Equity")
    }

    

    return {
        **state,
        "stock_info": {symbol: combined_data},
        "news_info": {symbol: news_info}
    }

# ---------------------------
# Node: Summarize
# ---------------------------
def summarize_node(state: AgentState) -> AgentState:  

   

    symbol = state["symbols"]  # already a string
    stock_info = state["stock_info"][symbol]
    news_info = state["news_info"][symbol] 
    
    
    prompt_template = PromptTemplate.from_template("""
    You are a financial analyst. Based on the stock data and news provided, write a detailed investment summary.

    Include:
    - Current price, range, volume
    - Sentiment summary from recent news
    - A markdown table with metrics like P/E ratio etc.
    - A final investment recommendation
    - A disclaimer

    Use markdown with headings and emojis.

    Stock Info:
    {stock_info}

    News:
    {news_info}
    """)
    
    llm = get_llm()

    summary_chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )

    summary = summary_chain.invoke({
        "stock_info": stock_info,
        "news_info": news_info
    })  
    


    return {**state, "summary": summary}
    

    
# ---------------------------
# Node: Compare Stocks
# ---------------------------



def compare_stocks_node(state: AgentState) -> AgentState:
    llm = get_llm()   
    
    symbols = state.get("compare_symbols", [])

    # Handle follow-up comparison like "Compare with Tesla"
    if len(symbols) == 1:
        last_symbols = state.get("last_symbols", [])
        if last_symbols:
            symbols = [last_symbols[0], symbols[0]]
        else:
            return {**state, "summary": "Please provide two stock symbols to compare."}

    if len(symbols) < 2:
        return {**state, "summary": "Please provide two stock symbols to compare."}

    symbol_a, symbol_b = symbols[0], symbols[1]

   

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
    financial_metrics = ["PE Ratio", "ROE", "ROA", "Debt/Equity", "Current Ratio"]
    ratio_table = "| Metric | " + symbol_a + " | " + symbol_b + " |\n|--------|" + "--------|"*2 + "\n"
    for metric in financial_metrics:
        val_a = ratios_a.get(metric)
        val_b = ratios_b.get(metric)
        ratio_table += f"| {metric} | {val_a} | {val_b} |\n"

    # Create structured LLM prompt
    prompt = f"""
    You are a professional stock market analyst comparing two companies using live financial data from FinancialModelingPrep and recent news headlines. Today's date is {today}.

    Compare {symbol_a} and {symbol_b} across these sections:

    1. 📊 **Financial Overview Table**  
    {ratio_table}  
    Source: FinancialModelingPrep (as of {today})

    2. 📌 **Company Strengths & Weaknesses**  
    Use your knowledge and recent news headlines to list 2-3 strengths and weaknesses for each company.

    3. 🔍 **Recent Trends & Outlook**  
    Summarize key business or market movements, earnings news, product launches, and growth plans.

    4. 📉 **News Sentiment Score (Past 7 Days)**  
    {symbol_a}: {sentiment_a}  
    {symbol_b}: {sentiment_b}

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
    builder.add_node("gather_stock_info", gather_stock_info_node)  # 🔁 NEW NODE
    builder.add_node("summarize", summarize_node)
    builder.add_node("compare_stocks", compare_stocks_node)

    builder.set_entry_point("analyze_query")

    builder.add_conditional_edges(
        "analyze_query",
        analyze_router,
        {
            "gather_stock_info": "gather_stock_info",           
            "compare_stocks": "compare_stocks"
        }
    )

    builder.add_edge("gather_stock_info", "summarize")
    builder.add_edge("summarize", END)
    builder.add_edge("compare_stocks", END)

    return builder.compile()


