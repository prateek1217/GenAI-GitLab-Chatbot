"""
CLI entrypoint for the GitLab Handbook Chatbot data pipeline.

Usage:
    python main.py scrape    # Crawl GitLab handbook + direction pages
    python main.py ingest    # Chunk, embed, and store in Pinecone
    python main.py linked    # Scrape linked pages (docs/about.gitlab.com) + ingest
    python main.py pipeline  # Run scrape then ingest in sequence
    streamlit run frontend.py  # Launch the Streamlit chat UI
"""
import sys


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "scrape":
        from backend.scraper import scrape_gitlab
        print("Starting GitLab handbook scrape...")
        scrape_gitlab()

    elif command == "ingest":
        from backend.ingest import ingest
        print("Starting ingestion into Pinecone...")
        ingest()

    elif command == "linked":
        from backend.scraper import scrape_linked_pages
        from backend.ingest import ingest
        print("=== Scraping linked pages ===")
        scrape_linked_pages()
        print("\n=== Ingesting new pages into Pinecone ===")
        ingest()

    elif command == "pipeline":
        from backend.scraper import scrape_gitlab
        from backend.ingest import ingest
        print("=== Step 1: Scraping ===")
        scrape_gitlab()
        print("\n=== Step 2: Ingesting ===")
        ingest()

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
