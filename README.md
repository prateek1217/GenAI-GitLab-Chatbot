<img width="1690" height="1390" alt="image" src="https://github.com/user-attachments/assets/551f9bbe-3d1a-41d7-9500-8504a0ce145c" />



# GenAI GitLab Chatbot

A RAG-based conversational AI assistant that answers questions about GitLab's public Handbook and Direction pages. Built with LangGraph, LangChain, Streamlit, RAG (Pinecone), Human in the Loop (HITL), Supabase and LLM.

---

## Features

- **RAG Pipeline** — GitLab handbook scraped, chunked, embedded, and stored in Pinecone vector DB
- **ReAct Agent** — LangGraph agent that reasons and decides when to use tools
- **Tools** — GitLab handbook retriever (Pinecone) + Tavily web search to find the most relevant data
- **Human-in-the-Loop** — Agent pauses and asks for approval before running web search
- **Relevance Gate** — Off-topic questions are blocked before reaching the agent
- **Streaming Responses** — Token-by-token streaming with live cursor
- **Conversation Memory** — Full conversation history persisted in Supabase (PostgreSQL)
- **Conversation Management** — Sidebar with conversation list, active indicator, and delete button
- **Thinking Indicator** — Animated dots while the AI is processing

---

## Concurrency & Async Design

The agent nodes (`relevance_gate_node`, `agent_node`) make network I/O calls — to Gemini, Pinecone, and Supabase — which involve waiting for external responses. Under high concurrency (e.g. 100 simultaneous users), these waits are the bottleneck.

**Why async/await would help here:**
Each node call blocks a thread while waiting for a network response. With `async def` nodes and `ainvoke()`/`astream()`, those waits become non-blocking — 100 users' LLM calls can be in-flight simultaneously instead of queued sequentially, reducing total response time from `100 × 2s = 200s` down to `~2s`.

**Why it's not implemented here:**
Streamlit runs on top of Tornado (an async web server) which already has a running event loop. Calling `asyncio.run()` inside Streamlit raises `RuntimeError: This event loop is already running`. Streamlit handles session-level concurrency via threads, which is sufficient for a demo and avoids this conflict.

**The production upgrade path:**
If this were deployed on an async server (FastAPI + Uvicorn), the change would be straightforward:
- `def relevance_gate_node` → `async def relevance_gate_node` with `await llm.ainvoke()`
- `def agent_node` → `async def agent_node` with `await llm.ainvoke()`
- `PostgresSaver` → `AsyncPostgresSaver` with `await psycopg.AsyncConnection.connect()`
- `graph.stream()` → `graph.astream()`

LangGraph natively supports all async equivalents — the architecture is async-ready, just not wired up due to Streamlit's event loop constraint.




## Architecture

```
User Input
    │
    ▼
Relevance Gate (Gemini) ──── Off-topic ──► Static refusal
    │
    ▼ GitLab-related
  Agent (Gemini + Tools)
    │
    ├── No tool calls ──────────────────► Final answer
    │
    └── Tool calls
            │
            ▼
      Tool Approval Node
            │
            ├── search_gitlab_handbook ──► Pinecone RAG ──► Agent ──► Answer
            │
            └── tavily_web_search
                      │
                      ▼
               Human Approval (Yes/No)
                      │
                      ├── Yes ──► Tavily Web Search ──► Agent ──► Answer
                      └── No  ──► Decline message
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash |
| Agent Framework | LangGraph |
| Vector DB | Pinecone |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (384 dims) |
| Web Search | Tavily |
| Web Scraping | Firecrawl |
| Conversation Memory | Supabase (PostgreSQL) via `langgraph-checkpoint-postgres` |
| Frontend | Streamlit |
| Package Manager | uv |

---

## Project Structure

```
GenAI_ChatBot/
├── frontend.py                  # Streamlit UI
├── main.py                      # CLI entrypoint for data pipeline
├── pyproject.toml               # Dependencies (managed by uv)
├── .env                         # API keys (not committed)
└── backend/
    ├── agent.py                 # LangGraph agent, nodes, graph wiring
    ├── scraper.py               # Firecrawl web scraper
    ├── ingest.py                # Chunking + Pinecone ingestion
    └── tools/
        ├── retriever.py         # Pinecone RAG tool
        └── web_search.py        # Tavily web search tool
```

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Accounts and API keys for:
  - [Google AI Studio](https://aistudio.google.com/) — Gemini API key
  - [Pinecone](https://www.pinecone.io/) — Vector DB
  - [Supabase](https://supabase.com/) — PostgreSQL (free tier works)
  - [Tavily](https://tavily.com/) — Web search API
  - [Firecrawl](https://www.firecrawl.dev/) — Web scraping (only needed to rebuild the data pipeline)

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/GenAI_ChatBot.git
cd GenAI_ChatBot
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=question-answer
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql://postgres.[project-ref]:[password]@[host]:5432/postgres
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

> **Supabase note:** Use the **Session Pooler** URL from your Supabase project dashboard (required for IPv4 networks). The username format must be `postgres.[project-ref]`.

### 4. Run the data pipeline

Scrape GitLab's handbook, direction, and community pages, then ingest them into Pinecone:

```bash
# Scrape + ingest in one command
uv run python main.py pipeline

# Optionally scrape linked pages (docs.gitlab.com, about.gitlab.com)
uv run python main.py linked
```

Individual commands:

```bash
uv run python main.py scrape    # Scrape only
uv run python main.py ingest    # Ingest only
```

### 5. Run the app

```bash
uv run streamlit run frontend.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Environment Variables Reference

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API key for Gemini |
| `PINECONE_API_KEY` | Pinecone API key |
| `INDEX_NAME` | Pinecone index name (default: `question-answer`) |
| `TAVILY_API_KEY` | Tavily search API key |
| `DATABASE_URL` | Supabase PostgreSQL connection string (Session Pooler URL) |
| `FIRECRAWL_API_KEY` | Firecrawl API key (data pipeline only) |
