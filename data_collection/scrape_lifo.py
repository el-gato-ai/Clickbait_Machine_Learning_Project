import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LifoMostPopularBot/1.0; +https://example.com)"
}

SECTIONS = {
    "ΕΙΔΗΣΕΙΣ",
    "ΘΕΜΑΤΑ",
    "GOOD LIFO",
    "PODCASTS",
    "VIDEOS",
    "CITY GUIDE",
}

def fetch_html(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def parse_mostpopular_page(html: str, known_titles: set) -> list[dict]:
    """
    Παίρνει HTML από μια σελίδα mostpopular και επιστρέφει unique records:
        - section
        - title
        - clickbait = 1
    Αποφεύγει duplicates χρησιμοποιώντας το known_titles set.
    """

    soup = BeautifulSoup(html, "html.parser")
    records = []

    for h2 in soup.find_all("h2"):
        section_text = h2.get_text(strip=True)
        if section_text not in SECTIONS:
            continue

        current_section = section_text

        ul = h2.find_next("ul")
        if not ul:
            continue

        for li in ul.find_all("li"):
            a = li.find("a")
            if not a:
                continue

            raw_text = a.get_text(" ", strip=True)

            title_part = raw_text.rsplit("|", maxsplit=1)[0].strip()
            title = title_part

            normalized = title.lower().strip()
            if normalized in known_titles:
                continue

            known_titles.add(normalized)

            records.append(
                {
                    "section": current_section,
                    "title": title,
                    "clickbait": 1,
                }
            )

    return records



def scrape_lifo_mostpopular():
    urls = [
        "https://www.lifo.gr/mostpopular/7days",
        "https://www.lifo.gr/mostpopular/today",
        'https://www.lifo.gr/mostpopular',
    ]

    all_records = []
    known_titles = set()

    for url in urls:
        print(f"[INFO] Fetching {url}")
        html = fetch_html(url)
        if html is None:
            continue

        page_records = parse_mostpopular_page(html, known_titles)
        print(f"[INFO] Added {len(page_records)} unique items from {url}")

        all_records.extend(page_records)

    if not all_records:
        print("[WARN] No records scraped.")
        return

    df = pd.DataFrame(all_records, columns=["section", "title", "clickbait"])
    df.to_csv("lifo_mostpopular_7days_today.csv", index=False, encoding="utf-8")
    print("[INFO] Saved lifo_mostpopular_7days_today.csv with unique titles.")


if __name__ == "__main__":
    scrape_lifo_mostpopular()
