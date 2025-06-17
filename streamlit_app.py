import streamlit as st
from graph_agent import build_stock_graph
from agent import extract_intent_and_symbols 
from llm import get_llm  
import json
import re

st.set_page_config(page_title="📈 Stock Market Research Agent", layout="wide")
st.title("📊 Stock Market Research Agent")
st.markdown("Ask about any public company or compare two (e.g.,`analyse nvidia` or `Compare Apple and Microsoft`)")

# Initialize graph and LLM
if "graph" not in st.session_state:
    st.session_state.graph = build_stock_graph()

if "llm" not in st.session_state:
    st.session_state.llm = get_llm()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Track context
if "last_symbols" not in st.session_state:
    st.session_state.last_symbols = []

if "last_intent" not in st.session_state:
    st.session_state.last_intent = ""

# Reset context button
if st.button("🔄 Reset Context"):
    st.session_state.last_symbols = []
    st.session_state.last_intent = ""
    st.success("Context reset!")

# Display current context
if st.session_state.last_symbols:
    st.markdown(f"🧠 **Context**: {', '.join(st.session_state.last_symbols)} | Intent: *{st.session_state.last_intent}*")

# Show past messages
for user_q, assistant_r in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        st.markdown(assistant_r)

# Input box for chat
user_query = st.chat_input("Ask a stock market question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Analyzing..."):

        # Extract intent + symbols using LLM
        extraction = extract_intent_and_symbols(st.session_state.llm, user_query)
        symbols = extraction.get("symbols", [])
        intent = extraction.get("intent", "")

        # 🧠 Handle follow-ups and fill gaps
        if intent == "compare_stocks" and len(symbols) == 1 and st.session_state.last_symbols:
            symbols = [st.session_state.last_symbols[0], symbols[0]]
        elif not symbols and st.session_state.last_symbols:
            symbols = st.session_state.last_symbols
            
        if not symbols:
            response = (
                "❗ I couldn't detect any company or stock symbols in your query.\n\n"
                "Try again using clear company names or stock tickers like `Apple`, `Tesla`, `AAPL`, `TSLA`, etc."
            )
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append((user_query, response))
            st.stop()
       

        # 🔁 Update memory
        st.session_state.last_symbols = symbols
        st.session_state.last_intent = intent

        # 🌐 Invoke your graph agent
        result = st.session_state.graph.invoke({
            "user_query": user_query,
            "user_intent": intent,
            "compare_symbols": symbols,
            "last_symbols": st.session_state.last_symbols,
            # Optional: prefill other inputs as empty
            "symbol": "",
            "stock_info": "",
            "news_info": "",
            "summary": "",
            "comparison_result": ""
        })

        response = result["summary"]

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append((user_query, response))
