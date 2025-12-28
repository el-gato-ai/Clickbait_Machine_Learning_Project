from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime, timedelta
from typing import List

from dotenv import find_dotenv, load_dotenv
from data_collection.prompt_eng import SearchModeLiteral, TopicLiteral, TOPICS
from data_collection.workflow import stream_all_topics_for_date


def run_agent(
    mode: SearchModeLiteral = "clickbait",
    start_date: str | None = None,
    end_date: str | None = None,
    country: str | None = None,
    language: str | None = None,
    topics: List[TopicLiteral] | None = None,
):
    """
    Execute the agent across all topics for the requested window, mode,
    country, and language. If topics is provided, only those topics run.
    Returns the responses keyed by topic.
    """
    responses_by_topic = stream_all_topics_for_date(
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        country=country,
        language=language,
        topics=topics,
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
        print(f"Distinct URLs from those returned: {len(unique_urls)}")
        print(f"Distinct (URL, title) pairs from those returned: {len(unique_pairs)}")

    return responses_by_topic


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect news articles per topic using Tavily + GPT. "
            "You can choose between 'news' (significant articles) "
            "and 'clickbait' (sensational headlines)."
        )
    )
    parser.add_argument(
        "--country",
        required=True,
        help="Target country/region for the run (e.g., France, Germany, United States).",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language to search/respond in (e.g., French, German, English).",
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date (YYYY-MM-DD). If omitted, defaults to ~30 days ago (UTC).",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date (YYYY-MM-DD). If omitted, defaults to today (UTC).",
    )
    parser.add_argument(
        "--mode",
        choices=["news", "clickbait"],
        default="clickbait",
        help=(
            "What kind of articles to collect: "
            "'news' for significant, balanced reporting (default), "
            "'clickbait' for sensational or exaggerated headlines."
        ),
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        choices=TOPICS,
        help="Optional list of topics to run (space separated). If omitted, all topics run.",
    )
    args = parser.parse_args()
    
    # Load environment variables once on import.
    _ = load_dotenv(find_dotenv())

    today_utc = datetime.utcnow().date()
    default_start = (today_utc - timedelta(days=30)).strftime("%Y-%m-%d")
    default_end = today_utc.strftime("%Y-%m-%d")

    start_date = args.start_date or default_start
    end_date = args.end_date or default_end

    run_agent(
        mode=args.mode,
        start_date=start_date,
        end_date=end_date,
        country=args.country,
        language=args.language,
        topics=args.topics,
    )


if __name__ == "__main__":
    main()
