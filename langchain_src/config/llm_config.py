# langchain_src/config/llm_config.py

import os
from langchain.chat_models import ChatOpenAI

def get_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
    """
    Returns a ChatOpenAI instance with your chosen model and temperature.
    Make sure OPENAI_API_KEY is set in environment or .env.
    """
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
