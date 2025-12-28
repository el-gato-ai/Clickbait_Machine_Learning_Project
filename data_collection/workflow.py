import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, after_model
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from data_collection.llms import get_chat_model, get_tools
from data_collection.prompt_eng import (
    NewsResponse,
    SearchModeLiteral,
    TopicLiteral,
    TOPICS,
    build_system_prompt,
    build_topic_query,
)

EXCEL_PATH = "../data/raw/custom/custom_news.xlsx"


def append_news_to_excel(
    response: NewsResponse,
    query: str,
    excel_path: str = EXCEL_PATH,
    country: Optional[str] = None,
    language: Optional[str] = None,
) -> int:
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
            "country": country,
            "language": language,
        }
        for item in response.items
    ]
    new_df = pd.DataFrame(rows)

    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
        existing_urls = set(existing_df["url"].astype(str).tolist()) if "url" in existing_df.columns else set()
    else:
        existing_df = None
        existing_urls = set()

    # Ensure the existing file has the new columns so concatenation succeeds.
    if existing_df is not None:
        for col in ["country", "language"]:
            if col not in existing_df.columns:
                existing_df[col] = None

    filtered_df = new_df[~new_df["url"].astype(str).isin(existing_urls)]

    if filtered_df.empty:
        print("[after_model] No new URLs to add (all duplicates).")
        return 0

    combined_df = (
        pd.concat([existing_df, filtered_df], ignore_index=True) if existing_df is not None else filtered_df
    )
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

    country = state.get("country")
    language = state.get("language")

    added = append_news_to_excel(structured, query, EXCEL_PATH, country=country, language=language)
    print(
        f"[after_model] Appended {added} new rows to {EXCEL_PATH} "
        f"(deduped by URL). country={country}, language={language}"
    )

    return None


def build_news_agent(country: Optional[str] = None, language: Optional[str] = None):
    """
    Build a fresh agent instance.
    Called per-topic so that state does not leak across topics.
    """
    return create_agent(
        model=get_chat_model(),
        tools=get_tools(country=country, language=language),
        system_prompt=build_system_prompt(country=country, language=language),
        response_format=NewsResponse,
        middleware=[save_news_after_model],
    )


def stream_news_agent(query: str, country: Optional[str] = None, language: Optional[str] = None) -> NewsResponse:
    """
    Stream agent progress to CLI for a single query.
    Excel append is handled automatically by @after_model middleware.
    """
    agent = build_news_agent(country=country, language=language)
    inputs = {
        "messages": [{"role": "user", "content": query}],
        "country": country,
        "language": language,
    }
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
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
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
    country: Optional[str] = None,
    language: Optional[str] = None,
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
            country=country,
            language=language,
        )
        response = stream_news_agent(query, country=country, language=language)
        results[topic] = response

    return results
