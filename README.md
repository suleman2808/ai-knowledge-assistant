# AI Knowledge Assistant — RAG + Multi-Agent Router

A customer-facing assistant for a service business (demo domain: a dental
clinic). An incoming message is classified by a router and dispatched to one
of three specialist agents:

| Agent | Responsibility |
| --- | --- |
| **Inquiry** | Answers questions about services, hours, pricing and policies using RAG over the clinic's own documents. Never answers from model knowledge. |
| **Booking** | Checks availability and creates appointments in Google Calendar. |
| **Complaint** | Logs complaints with a severity rating and escalates the serious ones. |

Every interaction is logged for analytics.

> **Status:** under construction. Step 1 of 10 complete (configuration and
> LLM abstraction layer).

## Stack

- **LangGraph** — agent orchestration as an explicit, inspectable graph
- **Groq** (`llama-3.3-70b-versatile`) — LLM inference, free tier
- **sentence-transformers** — embeddings computed locally, no API calls
- **ChromaDB** — persisted local vector store
- **FastAPI** — HTTP layer with auto-generated interactive docs
- **SQLite** — analytics logging
- **Google Calendar API** — real bookings, with an in-memory fallback

No paid services. Every dependency runs on a free tier or locally.

## Setup

Requires Python 3.11+ (developed on 3.13).

```bash
git clone https://github.com/suleman2808/ai-knowledge-assistant.git
cd ai-knowledge-assistant

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Add a free Groq API key (from [console.groq.com](https://console.groq.com))
to `.env`, then verify the setup:

```bash
python -m scripts.smoke_test
```

## Project layout

```
app/
  config.py          Typed configuration, loaded once from .env
  llm.py             The only module that talks to an LLM provider
  graph/             LangGraph state, router and agent nodes
  rag/               Ingestion, chunking, embeddings, retrieval
  integrations/      Google Calendar, analytics
data/documents/      Source documents for the knowledge base
scripts/             CLI entry points
tests/               pytest suite
```

## Design notes

**Why a single LLM module.** Nothing outside `app/llm.py` imports a provider
SDK. Switching from Groq to OpenAI, Anthropic or a local model means
rewriting one function. Retry policy, backoff and error translation live
there too, so failure handling is consistent rather than duplicated at every
call site.

**Why two models.** Intent classification into three buckets does not need a
70B model. The router uses `llama-3.1-8b-instant`, which is faster and
lighter on the free tier's rate limit; the agents use the larger model where
answer quality actually matters.

**Why local embeddings.** Embedding a few hundred chunks through an API
means a key, a bill and a rate limit. `all-MiniLM-L6-v2` runs on CPU in
seconds and makes the project work offline.

More design notes are added with each step.
