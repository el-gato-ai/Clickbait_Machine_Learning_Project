from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Topic labels used across the agent.
TopicLiteral = Literal[
    "Politics",
    "Economy",
    "Sports",
    "Society",
    "Technology",
    "Culture",
    "Fashion",
]

TOPICS: List[TopicLiteral] = [
    "Politics",
    "Economy",
    "Sports",
    "Society",
    "Technology",
    "Culture",
    "Fashion",
]

# Toggle between balanced news and clickbait-seeking mode.
SearchModeLiteral = Literal["news", "clickbait"]


class NewsItem(BaseModel):
    """Single news item from a news site."""

    url: str = Field(description="Full URL of the article")
    title: str = Field(description="Title of the article")
    description: str = Field(description="Short summary or description (2-3 sentences in English)")
    topic: TopicLiteral = Field(
        description=(
            "Topic label - exactly one of: "
            "Politics, Economy, Sports, Society, Technology, Culture, Fashion"
        )
    )


class NewsResponse(BaseModel):
    """List of extracted news items."""

    items: List[NewsItem] = Field(description="List of relevant news items for the given query")


def build_system_prompt(country: Optional[str] = None, language: Optional[str] = None) -> str:
    """
    Construct a system prompt that steers the agent toward the requested
    country and search language without hard-coding a single geography.
    """
    country_text = country or "the requested country or region"
    language_text = language or "the requested language"

    return f"""
You are a web-search agent that finds news articles.

BEHAVIOR:
    - Always use the TavilySearch tool to answer queries.
    - Focus on news sources from {country_text}; prefer outlets based there.
    - Write all responses in {language_text} unless explicitly instructed otherwise.

OUTPUT STRUCTURE:
    - Always return a valid `NewsResponse`:
        - items: list of NewsItem
    - Each NewsItem must contain:
        - url: article URL
        - title: article title
        - description: 2-3 sentence summary in the requested language
        - topic: EXACTLY one of:
            "Politics", "Economy", "Sports", "Society",
            "Technology", "Culture", "Fashion"

TOPIC LABELING:
    - Pick the topic label deterministically:
        - Politics   -> political news, government, parties, elections, diplomacy
        - Economy    -> macroeconomy, markets, companies, finance, inflation, jobs
        - Sports     -> football, basketball, athletes, matches, transfers, results
        - Society    -> social issues, crime, education, health system, daily life
        - Technology -> AI, startups, gadgets, software, digital policy, innovation
        - Culture    -> cinema, music, theatre, arts, literature, festivals
        - Fashion    -> fashion shows, trends, style, designers, brands
    - Do NOT invent other categories.

TIME FILTERING:
    - When the user query explicitly mentions a date or date range
        (for example "between 2025-01-01 and 2025-01-07"),
        you MUST:
            - pass these as `start_date` and `end_date` parameters
            in the TavilySearch tool call (format: YYYY-MM-DD),
            - and also respect them in your reasoning.
    - NEVER use the `time_range` parameter in TavilySearch tool calls.
        If you pass `start_date` or `end_date`, `time_range` MUST be omitted.

INSTRUCTIONS:
    - Use `results[].title`, `results[].content`, and `results[].url` from TavilySearch results.
    - Honor the country and language context in your tool calls and reasoning.
    - Skip generic homepages; prefer individual articles.
    - If the query is broad (e.g. "latest news"), return 5-8 diverse, representative articles.
    - Dont do a thousand search iterations; be concise and efficient. Max 4-5 tool calls.
"""


def build_topic_query(
    topic: TopicLiteral,
    start_date: str | None = None,
    end_date: str | None = None,
    mode: SearchModeLiteral = "news",
    country: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Build a natural language query for a given topic, date range,
    desired article style (news vs clickbait), and optional
    country/language targeting.
    Dates should be in YYYY-MM-DD if provided.
    """
    if mode == "clickbait":
        base = (
            f"Find news articles with CLICKBAIT, sensational, exaggerated, "
            f"or emotionally charged headlines about '{topic}' from news websites. "
            "Prefer provocative, dramatic, or curiosity-inducing headlines even if the story "
            "itself is not especially important."
        )
    else:
        base = (
            f"Find important and significant news articles about '{topic}' "
            "from reputable news websites. Avoid clickbait; prefer balanced, "
            "informative reporting on meaningful events."
        )

    if country:
        base += f" Focus on sources based in {country} and events relevant to that country."

    if language:
        base += f" The articles and search results should be in {language}."

    if start_date and end_date:
        base += (
            f" Only include articles published between {start_date} and {end_date} "
            f"(publish date, inclusive)."
        )
    elif start_date and not end_date:
        base += (
            f" Only include articles published on or after {start_date} "
            f"(publish date)."
        )
    else:
        base += " Focus on very recent news (last few months)."

    base += (
        " Return multiple distinct articles (ideally 5-8), not updates on the same one."
    )

    return base
