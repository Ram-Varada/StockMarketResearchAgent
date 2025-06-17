import streamlit as st
from graph_agent import build_stock_graph

st.set_page_config(page_title="📈 Stock Market Research Agent", layout="wide")
st.title("📊 Stock Market Research Agent")

st.markdown("Ask me about any public company or compare two (e.g., `Compare Apple and Microsoft`)")

# Initialize graph once
graph = build_stock_graph()

user_query = st.text_input("Your Query", "")

if user_query:
    with st.spinner("Analyzing..."):
        result = graph.invoke({
            "user_query": user_query,
            "symbol": "",
            "user_intent": "", 
            "stock_info": "",
            "news_info": "",
            "summary": "",
            "compare_symbols": [],
            "comparison_result": ""
        })

        st.markdown(result["summary"])
