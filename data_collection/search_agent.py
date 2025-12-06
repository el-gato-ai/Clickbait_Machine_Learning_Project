import os
import argparse
from typing import List, Literal, Dict
from datetime import datetime, timedelta

import pandas as pd
from pydantic import BaseModel, Field

from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain.agents import create_agent
from langchain.agents.middleware import after_model, AgentState
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage

_ = load_dotenv(find_dotenv())


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

SearchModeLiteral = Literal["news", "clickbait"]


class NewsItem(BaseModel):
    """Single news item from a Greek news site."""
    url: str = Field(description="Full URL of the article")
    title: str = Field(description="Title of the article")
    description: str = Field(
        description="Short summary or description (2-3 sentences in English)"
    )
    topic: TopicLiteral = Field(
        description=(
            "Topic label - exactly one of: "
            "Politics, Economy, Sports, Society, Technology, Culture, Fashion"
        )
    )


class NewsResponse(BaseModel):
    """List of extracted news items."""
    items: List[NewsItem] = Field(description="List of relevant news items for the given query")


search_tool = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    topic="news",
    time_range=None,
    include_answer=False,
    include_raw_content=False,
    include_domains=[
        "kathimerini.gr",
        "naftemporiki.gr",
        "news247.gr",
        "in.gr",
        "protothema.gr",
        "skai.gr",
        "efsyn.gr",
        "ertnews.gr",
    ],
)


SYSTEM_PROMPT = """
You are a web-search agent that finds Greek news articles.

BEHAVIOR:
    - Always use the TavilySearch tool to answer queries.
    - Focus exclusively on Greek news sites (politics, economy, society, sports, tech, culture, fashion).
    - Write all responses in English.

OUTPUT STRUCTURE:
    - Always return a valid `NewsResponse`:
        - items: list of NewsItem
    - Each NewsItem must contain:
        - url: article URL
        - title: article title
        - description: 2-3 sentence summary in English
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
    - Skip generic homepages; prefer individual articles.
    - If the query is broad (e.g. "latest news"), return 5-8 diverse, representative articles.
    - Dont do a thousand search iterations; be concise and efficient. Max 4-5 tool calls.
"""


EXCEL_PATH = "../data/raw/custom/greek_news.xlsx"


def append_news_to_excel(response: NewsResponse, query: str, excel_path = EXCEL_PATH) -> int:
    """
    Append news items to Excel, deduplicating by URL.
    Returns the number of NEW rows appended.
    """
    now = datetime.utcnow().isoformat()

    rows = [
        {
            "timestamp_utc": now,
            "query": query,
            "url": item.url,
            "title": item.title,
            "description": item.description,
            "topic": item.topic,
        }
        for item in response.items
    ]
    new_df = pd.DataFrame(rows)

    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
        if "url" in existing_df.columns:
            existing_urls = set(existing_df["url"].astype(str).tolist())
        else:
            existing_urls = set()
    else:
        existing_df = None
        existing_urls = set()

    filtered_df = new_df[~new_df["url"].astype(str).isin(existing_urls)]

    if filtered_df.empty:
        print("[after_model] No new URLs to add (all duplicates).")
        return 0

    if existing_df is not None:
        combined_df = pd.concat([existing_df, filtered_df], ignore_index=True)
    else:
        combined_df = filtered_df

    combined_df.to_excel(excel_path, index=False)
    return len(filtered_df)


@after_model
def save_news_after_model(state: AgentState, runtime: Runtime) -> dict | None:
    """
    Runs after each model call.
    If a structured_response (NewsResponse) is present, append to Excel.
    """
    structured = state.get("structured_response")
    if structured is None:
        return None

    items = getattr(structured, "items", None)
    if not items:
        return None

    # Find last user query
    query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    added = append_news_to_excel(structured, query, EXCEL_PATH)
    print(f"[after_model] Appended {added} new rows to {EXCEL_PATH} (deduped by URL).")

    return None


def build_news_agent():
    """
    Build a fresh agent instance.
    Called per-topic so that state does not leak across topics.
    """
    return create_agent(
        model=ChatOpenAI(
            model="gpt-5-nano-2025-08-07",
            temperature=0,
            reasoning_effort="low",
        ),
        tools=[search_tool],
        system_prompt=SYSTEM_PROMPT,
        response_format=NewsResponse,
        middleware=[save_news_after_model],
    )


def build_topic_query(
    topic: TopicLiteral,
    start_date: str | None = None,
    end_date: str | None = None,
    mode: SearchModeLiteral = "news",
) -> str:
    """
    Build a natural language query for a given topic, date range,
    and desired article style (news vs clickbait).
    Dates should be in YYYY-MM-DD if provided.
    """
    if mode == "clickbait":
        base = (
            f"Find Greek news articles with CLICKBAIT, sensational, exaggerated, "
            f"or emotionally charged headlines about '{topic}' from Greek news websites. "
            "Prefer provocative, dramatic, or curiosity-inducing headlines even if the story "
            "itself is not especially important."
        )
    else:
        base = (
            f"Find important and significant Greek news articles about '{topic}' "
            "from reputable Greek news websites. Avoid clickbait; prefer balanced, "
            "informative reporting on meaningful events."
        )

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
        base += " Focus on very recent news (last few days)."

    base += (
        " Return multiple distinct articles (ideally 5-8), not updates on the same one."
    )

    return base


def stream_greek_news_agent(query: str) -> NewsResponse:
    """
    Stream agent progress to CLI for a single query.
    Excel append is handled automatically by @after_model middleware.
    """
    agent = build_news_agent()
    inputs = {"messages": [{"role": "user", "content": query}]}
    last_state = None

    print(f"\n=== Running agent for query: {query!r} ===\n")

    for state in agent.stream(inputs, stream_mode="values"):
        last_state = state
        messages = state.get("messages", [])
        if not messages:
            continue

        msg = messages[-1]
        role = getattr(msg, "role", msg.__class__.__name__)
        content = getattr(msg, "content", "")

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        print(f"[{role.upper()}] {content}\n" + "-" * 60)

    if last_state is None or "structured_response" not in last_state:
        raise RuntimeError("Agent did not produce a structured_response")

    return last_state["structured_response"]


def stream_all_topics_for_date(
    start_date: str | None = None,
    end_date: str | None = None,
    topics: List[TopicLiteral] | None = None,
    mode: SearchModeLiteral = "news",
) -> Dict[TopicLiteral, NewsResponse]:
    """
    For each topic, build a dynamic query including the date range,
    stream the agent, and collect the NewsResponse per topic.

    Excel appending happens automatically via @after_model.
    """
    if topics is None:
        topics = TOPICS

    results: Dict[TopicLiteral, NewsResponse] = {}

    for topic in topics:
        print("\n" + "=" * 80)
        print(f"### TOPIC: {topic} ###")
        print("=" * 80)

        query = build_topic_query(
            topic,
            start_date=start_date,
            end_date=end_date,
            mode=mode,
        )
        response = stream_greek_news_agent(query)
        results[topic] = response

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Collect Greek news articles per topic using Tavily + GPT. "
            "You can choose between 'news' (significant articles) "
            "and 'clickbait' (sensational headlines)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["news", "clickbait"],
        default="news",
        help=(
            "What kind of articles to collect: "
            "'news' for significant, balanced reporting (default), "
            "'clickbait' for sensational or exaggerated headlines."
        ),
    )
    args = parser.parse_args()

    today_utc = datetime.utcnow().date()
    start_date = (today_utc - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = today_utc.strftime("%Y-%m-%d")

    responses_by_topic = stream_all_topics_for_date(
        start_date=start_date,
        end_date=end_date,
        mode=args.mode,
    )

    print("\n=== Summary of collected articles ===")
    overall_url_title_pairs = set()

    for topic, resp in responses_by_topic.items():
        total_items = len(resp.items)
        unique_urls = {item.url for item in resp.items}
        unique_pairs = {(item.url, item.title) for item in resp.items}
        overall_url_title_pairs.update(unique_pairs)

        print(f"\n##### {topic} #####")
        print(f"Total articles returned: {total_items}")
        print(f"Distinct URLs in this topic: {len(unique_urls)}")
        print(
            f"Distinct (URL, title) pairs in this topic: "
            f"{len(unique_pairs)}"
        )

    print(
        "\n=== Overall distinct (URL, title) pairs across all topics "
        f"this run: {len(overall_url_title_pairs)} ==="
    )
