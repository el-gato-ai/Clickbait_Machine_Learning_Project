import os

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


def get_chat_model(model="gpt-5-nano-2025-08-07") -> ChatOpenAI:
    """Return the configured OpenAI chat model for the agent."""
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_effort="low",
    )


def get_tools():
    """
    Build the TavilySearch tool for a run, nudging toward a country via domains.
    Language is intentionally handled in the query/prompt to avoid relying on
    optional API params that may not be available.
    """
    return [
        TavilySearch(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5,
            topic="news",
            time_range=None,
            include_answer=False,
            include_raw_content=False,
        )
    ]
