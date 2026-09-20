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

> **Status:** under construction. Steps 1–4 of 10 complete (configuration,
> LLM abstraction, ingestion, retrieval, the three specialist agents).

## Stack

- **LangGraph** — agent orchestration as an explicit, inspectable graph
- **Groq** (`openai/gpt-oss-120b`) — LLM inference, free tier
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

If a model name has been retired since this was written, list what your key
can actually reach and update `.env` accordingly:

```bash
python -m scripts.list_models
```

## Building the knowledge base

```bash
python -m scripts.ingest
```

This chunks the documents in `data/documents/`, embeds them locally and
writes them to ChromaDB. The first run downloads the embedding model
(~90 MB); later runs take a few seconds. The demo corpus is nine documents
covering pricing, insurance, policies, clinical aftercare, emergencies,
staff and an FAQ — 74 chunks in total.

Inspect chunking without embedding anything:

```bash
python -m scripts.inspect_chunks --show 2
```

## Querying the knowledge base

```bash
python -m scripts.search "how much is a root canal"
python -m scripts.search --calibrate
```

## Project layout

```
app/
  config.py          Typed configuration, loaded once from .env
  llm.py             The only module that talks to an LLM provider
  agents/            The three specialists, each a plain function
  prompts/           Prompt templates as .md files
  graph/             LangGraph state, router and wiring
  rag/               Ingestion, chunking, embeddings, retrieval
  integrations/      Calendar, analytics
data/documents/      Source documents for the knowledge base
scripts/             CLI entry points
tests/               pytest suite
```

Run any agent on its own, without the graph or the API:

```bash
python -m scripts.try_agent inquiry --suite
python -m scripts.try_agent booking "book a cleaning next Tuesday at 2pm"
python -m scripts.try_agent complaint "I was charged twice"
```

## Design notes

**Why a single LLM module.** Nothing outside `app/llm.py` imports a provider
SDK. Switching from Groq to OpenAI, Anthropic or a local model means
rewriting one function. Retry policy, backoff and error translation live
there too, so failure handling is consistent rather than duplicated at every
call site.

**Why two models.** Intent classification into three buckets does not need
the large model. The router uses `openai/gpt-oss-20b` at low reasoning
effort — faster and lighter on the free tier's rate limit; the agents use
`openai/gpt-oss-120b`, where answer quality actually matters.

**Reasoning-model token budgets.** The `gpt-oss` models spend hidden
reasoning tokens from the same `max_tokens` budget as the visible reply. Set
the cap too low and the API returns success with an *empty* string rather
than an error. `app/llm.py` detects that case and raises an explanatory
error naming the real cause, because an empty reply is otherwise extremely
hard to diagnose.

**Why local embeddings.** Embedding a few hundred chunks through an API
means a key, a bill and a rate limit. `all-MiniLM-L6-v2` runs on CPU in
seconds and makes the project work offline.

**Chunking strategy.** Chunks are split on markdown headings rather than a
fixed character count, so each chunk is a unit the author already decided
was coherent. Three consequences:

- *Tables are never separated from their header row.* A split price table
  leaves rows whose numbers survive but whose meaning does not. When a
  table genuinely exceeds the size limit, it is split row-wise with the
  header repeated in every part.
- *The heading path is prepended to the embedded text.* A section reading
  only "A fee of $50 applies" embeds poorly on its own; prefixed with
  "Appointment Policy > Cancellation Policy" it retrieves correctly. The
  prefix is added to the embedded form only — the stored text stays clean.
- *Short sections merge only into siblings.* Merging a brief section into
  whatever happened to precede it can file it under an unrelated parent,
  so a complaints question gets answered by text cited as "Payment >
  Refunds". A test covers this; it caught the bug during development.

Size-based splitting is the fallback, not the strategy, and it splits on
paragraph then sentence boundaries with a 150-character overlap.

**How retrieval failures are handled.** `retrieve()` never raises for an
expected condition. It returns a `RetrievalResult` carrying a status the
agent branches on: `OK`, `LOW_CONFIDENCE` (matches exist but all score
below the threshold), `NO_MATCH`, `EMPTY_INDEX` (nothing ingested — an
operator error, distinct from a user one) and `UNAVAILABLE` (the store or
model failed). Exceptions are reserved for genuine bugs, so no ordinary
production condition can produce a stack trace in front of a customer.

**The threshold is necessary but not sufficient — measured, not assumed.**
`scripts/search.py --calibrate` scores twelve questions the documents
answer against six they do not. The groups *overlap*:

| Query | Score | In scope? |
| --- | --- | --- |
| "how long do fillings last" | 0.835 | yes |
| "what are your opening hours on Saturday" | 0.726 | yes |
| **"what time does the cinema open"** | **0.422** | **no** |
| "who do I speak to about a billing problem" | 0.384 | yes |
| "is there parking at the clinic" | 0.380 | yes |
| "what is the capital of France" | 0.051 | no |

No single threshold separates them. The cinema question outscores two
legitimate ones and retrieves the clinic's opening-hours table, which a
naive agent would happily answer from.

The response is a two-stage defence rather than a better number:

1. The threshold (0.30) rejects the obviously unrelated.
2. The Inquiry Agent's prompt requires that the retrieved context actually
   answers *this* question, and refuses when it does not.

Stage two is what catches the cinema case. The longer-term fixes — hybrid
BM25 plus vector search, or a cross-encoder reranker — are deferred until
the system works end to end, since both are tuning rather than
architecture.

**Agents are plain functions, not graph nodes.** Each agent takes a
message and returns an `AgentResponse`. None of them imports LangGraph,
FastAPI or another agent, so each is testable alone and step 5 wires them
into a graph without changing agent code. All 52 tests run in 0.33s
because every LLM call is stubbed — what is tested is the logic around
the model, not the model.

**Prompts are markdown files, not string literals.** The person who should
approve what a clinic says to patients does not read Python. Editing a
prompt produces a readable diff, and tuning becomes a content change
rather than a code change. Placeholders use `{{name}}` because prompts
contain literal JSON braces that `str.format` would choke on.

**The Booking Agent uses the LLM for one job only.** Extraction — pulling
a service, a date and a phone number out of free text, including relative
dates like "next Tuesday". Everything after that is deterministic code:
whether that time is inside opening hours, whether it clashes, what the
appointment length should be. The model cannot see the calendar, and a
confidently double-booked appointment is worse than no booking. Extracted
fields are validated before use: a date in the past is treated as missing
rather than silently shifted, because it means the model mis-resolved a
relative reference.

**Complaint severity is assessed separately from the reply.** Asking one
call to both judge severity and write an apology biases the judgement —
a model composing a soothing response mirrors the patient's tone, so an
angry message about a magazine outranks a calm report of a clinical
injury. Assessment runs first, as JSON, with no audience.

**Escalation is decided in code, never by the model.** The model's
`requires_escalation` is one input. Any reported harm, any mention of
legal action, and any high or critical severity escalates regardless of
what the model concluded — the rules override upward, never downward. A
missed escalation is expensive; a false positive costs one person one
email. During testing a double-billing complaint with three ignored phone
calls was rated `high` and *not* escalated because the model said so;
that is now impossible, and a test enforces it.

**A complaint is logged before the reply is generated.** If reply
generation fails the complaint still exists. If assessment itself fails,
the complaint is recorded at high severity and escalated anyway. The
record carries regulatory weight; the reply is courtesy.

**Invisible characters are stripped at the boundary.** The model returned
a phone number containing a soft hyphen — it rendered correctly and broke
when copied. `AgentResponse` normalises every answer, so no agent can
forget.

**Why cosine distance is set explicitly.** Chroma defaults to squared L2.
With normalised vectors the ranking would be the same, but the distance
range differs, which would make a fixed relevance threshold meaningless.
The threshold is what lets the Inquiry Agent say "I don't know" instead of
answering from a weak match.

More design notes are added with each step.
