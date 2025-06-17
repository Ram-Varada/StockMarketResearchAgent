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



def extract_intent_and_symbols(llm: BaseLanguageModel, query: str) -> Dict[str, Any]:
    prompt = f"""
    You are an intelligent assistant helping categorize user queries related to stock market analysis.

    Given the following query, identify:
    1. The user's intent - one of: "gather_stock_info"", or "compare_stocks".
    2. The official stock **ticker symbols** (like "AAPL", "MSFT") mentioned in the query.

    Return your answer as a JSON object with:
    - "intent": one of the valid intent values.
    - "symbols": a list of **ticker symbols** (1 or 2 max).

    Respond ONLY with the JSON block. No extra explanation.

    Query: "{query}"
    """


    response = llm.invoke(prompt)
    content = response.content

    

    # Extract JSON block inside ```json ... ```
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not json_match:
        # fallback: try to parse the whole content anyway (in case no backticks)
        json_str = content.strip()
    else:
        json_str = json_match.group(1)

    try:
        result = json.loads(json_str)
        
        intent = result.get("intent", "both")
        symbols = result.get("symbols", [])
        return {"intent": intent, "symbols": symbols}
    except Exception as e:
        print("[extract_intent_and_symbols] Failed to parse JSON. Using fallback.", e)
        return {"intent": "both", "symbols": [query.strip()]}




