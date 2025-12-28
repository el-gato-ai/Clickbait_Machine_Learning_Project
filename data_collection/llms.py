import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Load environment variables once on import.
_ = load_dotenv(find_dotenv())

# Representative domains by country to bias results without hard-coding a single geography.
COUNTRY_DOMAINS = {
    "greece": [
        "kathimerini.gr",
        "naftemporiki.gr",
        "news247.gr",
        "in.gr",
        "protothema.gr",
        "skai.gr",
        "efsyn.gr",
        "ertnews.gr",
    ],
    "france": [
        "lemonde.fr",
        "lefigaro.fr",
        "ouest-france.fr",
        "liberation.fr",
    ],
    "germany": [
        "spiegel.de",
        "zeit.de",
        "faz.net",
        "sueddeutsche.de",
    ],
    "united states": [
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "cnn.com",
        "reuters.com",
    ],
}


def _domains_for_country(country: str | None):
    if not country:
        return None
    return COUNTRY_DOMAINS.get(country.lower())


def get_chat_model(model="gpt-4.1-nano-2025-04-14") -> ChatOpenAI:
    """Return the configured OpenAI chat model for the agent."""
    return ChatOpenAI(
        model=model,
        temperature=0,
        reasoning_effort="low",
    )


def get_tools(country: str | None = None, language: str | None = None):
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
            include_domains=_domains_for_country(country),
        )
    ]
