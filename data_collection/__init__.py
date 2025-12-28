from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime, timedelta

from data_collection.prompt_eng import SearchModeLiteral
from data_collection.workflow import stream_all_topics_for_date


def run_agent(
    mode: SearchModeLiteral = "news",
    start_date: str | None = None,
    end_date: str | None = None,
    country: str | None = None,
    language: str | None = None,
):
    """
    Execute the agent across all topics for the requested window, mode,
    country, and language.
    Returns the responses keyed by topic.
    """
    responses_by_topic = stream_all_topics_for_date(
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        country=country,
        language=language,
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
        print(f"Distinct (URL, title) pairs in this topic: {len(unique_pairs)}")

    print(
        "\n=== Overall distinct (URL, title) pairs across all topics "
        f"this run: {len(overall_url_title_pairs)} ==="
    )

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

    run_agent(
        mode=args.mode,
        start_date=start_date,
        end_date=end_date,
        country=args.country,
        language=args.language,
    )


if __name__ == "__main__":
    main()
