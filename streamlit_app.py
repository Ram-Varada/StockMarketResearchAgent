import streamlit as st
from graph_agent import build_stock_graph

st.set_page_config(page_title="📈 Stock Market Research Agent", layout="wide")
st.title("📊 Stock Market Research Agent")
st.markdown("Ask me about any public company or compare two (e.g., `Compare Apple and Microsoft`)")

# Initialize graph once
if "graph" not in st.session_state:
    st.session_state.graph = build_stock_graph()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for user_query, response in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        st.markdown(response)

# Persistent input box
user_query = st.chat_input("Ask a stock market question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Analyzing..."):
        result = st.session_state.graph.invoke({
            "user_query": user_query,
            "symbol": "",
            "user_intent": "", 
            "stock_info": "",
            "news_info": "",
            "summary": "",
            "compare_symbols": [],
            "comparison_result": ""
        })
        response = result["summary"]

    with st.chat_message("assistant"):
        st.markdown(response)

    # Append to session history
    st.session_state.chat_history.append((user_query, response))
