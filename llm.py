from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models.llms import LLM
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROK_API_KEY = os.getenv("groq_api_key")

def get_llm():   
    
    llm=ChatGroq(groq_api_key=GROK_API_KEY,model_name="Gemma2-9b-It")
    return llm
