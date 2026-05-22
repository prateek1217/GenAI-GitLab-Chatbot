import os
import re
import json
import time
from firecrawl import Firecrawl
from firecrawl.v2.types import ScrapeOptions
from dotenv import load_dotenv

load_dotenv()

SEED_URLS = [
    "https://handbook.gitlab.com/handbook/",
    "https://handbook.gitlab.com/direction/",
    "https://about.gitlab.com/community/",
]

# Domains to follow when extracting linked pages
LINKED_DOMAINS = [
    "docs.gitlab.com",
    "about.gitlab.com",
]

OUTPUT_FILE = "backend/scraped_data.json"


def extract_linked_urls(scraped_file: str = OUTPUT_FILE) -> list[str]:
    """Return unique GitLab-domain URLs found in scraped markdown that haven't been scraped yet."""
    try:
        with open(scraped_file, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return []

    already_scraped = {page.get("url", "") for page in raw}
    url_pattern = re.compile(r'https?://[^\s\)\]"\'<>]+')
    found: set[str] = set()

    for page in raw:
        for url in url_pattern.findall(page.get("markdown", "")):
            url = url.rstrip(".,;:!?)")
            if not any(d in url for d in LINKED_DOMAINS):
                continue
            if url in already_scraped:
                continue
            if re.search(r'[\s`{]', url) or "/blog/" in url:
                continue
            found.add(url)

    urls = sorted(found)
    print(f"Found {len(urls)} new linked pages to scrape")
    return urls


def scrape_linked_pages(scraped_file: str = OUTPUT_FILE) -> list[dict]:
    """Scrape individual pages linked from the already-scraped handbook content."""
    app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    linked_urls = extract_linked_urls(scraped_file)
    if not linked_urls:
        print("No new linked pages found.")
        return []

    new_docs = []
    for i, url in enumerate(linked_urls, 1):
        print(f"  [{i}/{len(linked_urls)}] {url}")
        try:
            page = app.v2.scrape(
                url,
                formats=["markdown"],
                only_main_content=True,
                exclude_tags=["nav", "footer", "header", "script", "style"],
            )
            markdown = page.markdown or ""
            if len(markdown) >= 200:
                meta = page.metadata
                new_docs.append({
                    "markdown": markdown,
                    "url": (meta.url or url) if meta else url,
                    "metadata": {
                        "title": (meta.title or "") if meta else "",
                        "description": (meta.description or "") if meta else "",
                    },
                })
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(5)

    if new_docs:
        try:
            with open(scraped_file, encoding="utf-8") as f:
                existing = json.load(f)
        except FileNotFoundError:
            existing = []

        combined = existing + new_docs
        with open(scraped_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"\nAdded {len(new_docs)} new pages. Total: {len(combined)}")

    return new_docs


def scrape_gitlab(max_pages_per_url: int = 250) -> list[dict]:
    app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    all_docs = []

    scrape_opts = ScrapeOptions(
        formats=["markdown"],
        only_main_content=True,
        exclude_tags=["nav", "footer", "header", "script", "style"],
    )

    for url in SEED_URLS:
        print(f"\nCrawling: {url}")
        try:
            result = app.v2.crawl(
                url,
                limit=max_pages_per_url,
                max_discovery_depth=2,
                allow_external_links=False,
                scrape_options=scrape_opts,
            )

            # result is a CrawlJob with .data = List[Document]
            valid_pages = []
            for page in result.data:
                markdown = page.markdown or ""
                if len(markdown) < 200:
                    continue

                meta = page.metadata
                valid_pages.append({
                    "markdown": markdown,
                    "url": (meta.url or "") if meta else "",
                    "metadata": {
                        "title": (meta.title or "") if meta else "",
                        "description": (meta.description or "") if meta else "",
                    },
                })

            all_docs.extend(valid_pages)
            print(f"  Scraped {len(valid_pages)} valid pages from {url}")

        except Exception as e:
            print(f"  Error crawling {url}: {e}")
            time.sleep(5)

    print(f"\nTotal pages scraped: {len(all_docs)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_FILE}")
    return all_docs


if __name__ == "__main__":
    scrape_gitlab()
