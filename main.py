from graph_agent import build_stock_graph

from IPython.display import Image


if __name__ == "__main__":
   
    graph = build_stock_graph()
    
   
    
    # image_data = graph.get_graph().draw_mermaid_png()
    
    user_input = input("Ask me about any public company or compare two (e.g., `Compare Apple and Microsoft`)")

    # Save to file
    #with open("stock_graph.png", "wb") as f:
    #    f.write(image_data)

    result = graph.invoke({
        "user_query": user_input,
        "symbol": "",
        "user_intent": "both",  # placeholder
        "stock_data": "",
        "news_data": "",
        "summary": ""
    })

    print("\n🔍 Final Summary:\n")
    print(result["summary"])
