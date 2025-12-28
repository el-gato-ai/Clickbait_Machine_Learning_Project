import os
import re
from datetime import datetime
from pathlib import Path
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "custom" / "custom_news.xlsx"
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_text(value):
    if isinstance(value, str):
        return CONTROL_CHAR_RE.sub("", value)
    return value


def _sanitize_dataframe_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove illegal control characters from all object columns to keep Excel writes safe.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(_clean_text)
    return df


def append_news_to_excel(
    response: NewsResponse,
    query: str,
    excel_path: Path | str = EXCEL_PATH,
    country: Optional[str] = None,
    language: Optional[str] = None,
    topic: Optional[TopicLiteral] = None,
) -> int:
    """
    Append news items to Excel, deduplicating by URL + title.
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
            "topic": topic,
            "country": country,
            "language": language,
        }
        for item in response.items
    ]
    new_df = _sanitize_dataframe_text(pd.DataFrame(rows))

    # Ensure destination directory exists before reading/writing.
    excel_path = Path(excel_path)
    dest_dir = excel_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Load existing data if present to deduplicate.
    if excel_path.exists():
        existing_df = _sanitize_dataframe_text(pd.read_excel(excel_path))
        existing_urls = set(existing_df["url"].astype(str).tolist()) if "url" in existing_df.columns else set()
        if {"url", "title"}.issubset(existing_df.columns):
            existing_pairs = set(
                zip(existing_df["url"].astype(str).tolist(), existing_df["title"].astype(str).tolist())
            )
        else:
            existing_pairs = set()
    else:
        existing_df = None
        existing_pairs = set()
        existing_urls = set()

    # Deduplicate new data against existing by (url, title) pairs, falling back to url only.
    new_pairs = new_df.apply(lambda r: (str(r["url"]), str(r["title"])), axis=1)
    if existing_pairs:
        dedupe_mask = new_pairs.isin(existing_pairs)
    elif existing_urls:
        dedupe_mask = new_df["url"].astype(str).isin(existing_urls)
    else:
        dedupe_mask = pd.Series(False, index=new_df.index)

    # Filter to only truly new rows.
    filtered_df = new_df[~dedupe_mask]

    if filtered_df.empty:
        print("[after_model] No new URL+title combos to add (all duplicates).")
        return 0

    combined_df = (
        pd.concat([existing_df, filtered_df], ignore_index=True) if existing_df is not None else filtered_df
    )
    combined_df.to_excel(excel_path, index=False)
    return len(filtered_df)


def make_save_news_after_model(
    default_country: Optional[str] = None,
    default_language: Optional[str] = None,
    default_topic: Optional[TopicLiteral] = None,
):
    """
    Build an after_model middleware that carries country/language defaults
    from the current run (state is unreliable for these fields).
    """

    @after_model
    def save_news_after_model(state: AgentState, runtime: Runtime) -> dict | None:
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

        country = default_country
        language = default_language

        added = append_news_to_excel(
            structured,
            query,
            EXCEL_PATH,
            country=country,
            language=language,
            topic=default_topic,
        )
        print(
            f"[after_model] Appended {added} new rows to {EXCEL_PATH} "
            f"(deduped by URL + title, URL fallback if missing title). topic={default_topic}, country={country}, language={language}"
        )

        return None

    return save_news_after_model


def build_news_agent(
    country: Optional[str] = None,
    language: Optional[str] = None,
    topic: Optional[TopicLiteral] = None,
):
    """
    Build a fresh agent instance.
    Called per-topic so that state does not leak across topics.
    """
    return create_agent(
        model=get_chat_model(),
        tools=get_tools(),
        system_prompt=build_system_prompt(country=country, language=language),
        response_format=NewsResponse,
        middleware=[
            make_save_news_after_model(
                default_country=country,
                default_language=language,
                default_topic=topic,
            )
        ],
    )


def stream_news_agent(
    query: str,
    country: Optional[str] = None,
    language: Optional[str] = None,
    topic: Optional[TopicLiteral] = None,
) -> NewsResponse:
    """
    Stream agent progress to CLI for a single query.
    Excel append is handled automatically by @after_model middleware.
    """
    agent = build_news_agent(country=country, language=language, topic=topic)
    inputs = {
        "messages": [{"role": "user", "content": query}],
        "country": country,
        "language": language,
        "topic": topic,
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
        response = stream_news_agent(query, country=country, language=language, topic=topic)
        results[topic] = response

    return results
