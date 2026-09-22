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

> **Status:** under construction. Steps 1–7 of 10 complete (configuration,
> LLM abstraction, ingestion, retrieval, agents, router, Google Calendar,
> analytics).

```
                      ┌─────────┐
   message  ────────▶ │ router  │  keyword fast-path, else small LLM
                      └────┬────┘
           ┌───────────┬───┴───────┬───────────┐
           ▼           ▼           ▼           ▼
       ┌───────┐  ┌─────────┐ ┌──────────┐ ┌───────┐
       │booking│  │ inquiry │ │complaint │ │ other │
       └───┬───┘  └────┬────┘ └────┬─────┘ └───┬───┘
           └───────────┴─────┬─────┴───────────┘
                             ▼
                       ┌──────────┐
                       │ finalise │  secondary-intent note, turn record
                       └──────────┘
```

The real diagram is generated from the compiled graph — see
[docs/architecture.md](docs/architecture.md), regenerated with
`python -m scripts.draw_graph --write`.

## Stack

- **LangGraph** — agent orchestration as an explicit, inspectable graph
- **Groq** (`openai/gpt-oss-120b`) — LLM inference, free tier
- **all-MiniLM-L6-v2 on ONNX Runtime** — embeddings computed locally, no
  API calls, no PyTorch (`sentence-transformers` available as an option)
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

## Google Calendar (optional)

**Skip this and everything still works.** Without credentials the
Booking Agent uses an in-memory calendar that enforces real clash
detection, so a fresh clone runs end to end with no Google account.

To book into a real calendar:

1. [console.cloud.google.com](https://console.cloud.google.com/) → new project
2. APIs & Services → Library → **Google Calendar API** → Enable
3. OAuth consent screen → External → **add your own address as a test user**
4. Credentials → Create Credentials → OAuth client ID → **Desktop app**
5. Download the JSON as `credentials.json` in the project root (gitignored)

```bash
python -m scripts.google_auth          # authorise, once, in a browser
python -m scripts.google_auth --check  # verify and list free slots
python -m scripts.google_auth --revoke # back to the in-memory calendar
```

## Analytics

Every turn is recorded in SQLite (`data/analytics.db`, gitignored).

```bash
python -m scripts.seed_demo --reset     # realistic demo traffic, through the real graph
python -m scripts.analytics_report      # what patients asked, and how well it went
```

The report is organised around questions a clinic owner asks: what do
patients want, can the assistant answer them, is it creating work or
saving it, and is it fast. The most useful section is **knowledge gaps**
— questions the assistant had to decline, which are things patients
want to know that nobody has written down.

Talk to the assembled assistant:

```bash
python -m scripts.chat                     # interactive, keeps history
python -m scripts.chat --routing           # routing accuracy probe
python -m scripts.chat --scenario booking  # scripted multi-turn booking
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

**Contact details are redacted before anything reaches disk.** Analytics
needs *what* patients asked, not *who* asked. Phone numbers and emails
are stripped at the single write boundary, so no caller can forget.
Probing the redactor found it also mangled ISO dates and turned the
complaint reference `CMP-20260920-4F2A` into `CMP-[phone]-4F2A` —
destroying the one identifier staff trace a complaint by. Both are now
covered by tests. Names are not redacted; reliable name detection needs
an NER model, and that limitation is stated rather than hidden.

**Logging never breaks a conversation — except where it must.** A failed
turn write is logged and swallowed; an analytics outage is an
inconvenience, a chat that errors because of one is a defect. Complaints
are the exception: a failed complaint write is caught, held in memory,
logged at error level and flagged on the response, because a silently
lost complaint is the one failure this project treats as unacceptable.

**Logging lives in `run()`, not inside a graph node.** The graph stays
free of side effects beyond the agents' own, so it can be invoked in
tests or evaluations without polluting real numbers. The routing probe
passes `log=False`, and every test gets its own throwaway database via
an autouse fixture — before that fixture existed, the graph tests would
have written into the real analytics file.

**The first report found three retrieval bugs.** Of eight "knowledge
gaps" in the seeded data, only four were real. The other four had three
distinct causes: a bare brand name ("do you take cigna") scoring 0.23
because an embedding model gives proper nouns little weight; a correct
retrieval refused by an over-strict grounding check; and follow-up
questions ("is that for one surface?") retrieved without the
conversation that gives them meaning. Analytics doing its job on the
system that produces it.

**Why ONNX rather than PyTorch.** The project began on
`sentence-transformers`. Mid-build, Windows Smart App Control started
refusing PyTorch's unsigned `c10.dll` (`WinError 4551`), and every
inquiry failed. The system degraded as designed — patients were told
search was unavailable rather than shown a stack trace — but "disable an
OS security feature" is not an acceptable install step, and anyone
cloning onto a locked-down Windows machine would hit the same wall.

The fix changed the runtime, not the model: Chroma ships
`all-MiniLM-L6-v2` as an ONNX export on Microsoft's signed
`onnxruntime`. Before switching, the new vectors were compared against
the stored PyTorch ones for all 74 chunks — cosine similarity 1.000000
throughout — so the retrieval threshold calibrated earlier remains
valid. Side effects, all good: PyTorch (~1.5 GB) is no longer a required
dependency, and ingestion dropped from minutes to 8 seconds.
`sentence-transformers` remains available via `requirements-torch.txt`
and `EMBEDDING_BACKEND=sentence-transformers`, for models ONNX does not
cover.

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

**Timezones are converted at exactly one boundary.** The rest of the
project uses naive datetimes meaning clinic wall-clock time, which is
the right model for a business whose hours are "Monday 8am to 5pm"
regardless of where the patient is sitting. Google works in absolute
time. `app/integrations/google_calendar.py` is the only module that
converts, and it is tested across a DST boundary: 2pm is `-08:00` in
January and `-07:00` in July. An appointment written an hour out is
worse than no appointment, because the patient turns up.

**The OAuth consent flow lives in a script, not in the app.** It opens a
browser, which must never happen inside an HTTP request — the server
would block on a dialogue nobody can see. At runtime the backend only
*loads* a stored token, refreshing it silently when expired; if there is
no usable token it raises, and `get_calendar()` falls back to the
in-memory calendar. The scope requested is `calendar.events`, not
`calendar`, so the consent screen does not ask for permission to delete
entire calendars.

**A clash is re-checked immediately before writing.** Google has no
conditional insert, so a check cannot be atomic with the write. A
conversation takes seconds; re-checking narrows the race to
milliseconds. The remaining window is handled rather than ignored — the
Booking Agent treats a write-time clash as "offer alternatives", not as
an error.

**Events the organiser marked free, and cancelled events, do not block
the diary**, and recurring events are expanded with `singleEvents` —
without it a weekly staff meeting blocks only its first week and every
other Tuesday looks bookable.

**Why LangGraph rather than an if/else router.** The dispatch itself
*could* be four lines of `if`. What the graph buys is everything around
it: the structure is inspectable (the diagram is generated from the
compiled graph, so it cannot drift from what runs), state is explicit and
typed rather than whatever locals are in scope, each node is a pure
`state -> partial state` function testable without the graph, and adding
human handoff or a clarification loop means adding a node and an edge
rather than nesting another branch inside a function nobody wants to
touch. Honest version of the trade-off: for exactly three static
branches, if/else is simpler. Choosing a graph is a bet that the system
grows — and the moment a fourth intent appears, the if/else version
starts paying interest.

**The router has a keyword fast-path, and it is deliberately narrow.**
"hi" and "thanks" are among the commonest messages a public chat box
receives, and spending 400ms and an LLM call on them is waste — they
resolve in about 9ms. But the fast-path only matches *whole messages*,
never substrings, and only up to three words. "I want to cancel my
complaint about the cancellation fee" contains three trigger words and
means one thing; anything with real content goes to the model. Naive
keyword routers break precisely here, and a test covers it.

**Routing accuracy is measured, not assumed.** `--routing` scores 18
hand-labelled messages including the cases designed to be hard: policy
question versus diary change ("what is your cancellation policy" is an
inquiry, "I need to cancel Tuesday" is a booking), an angry question
that is still a question, and a message whose only content is
"Sarah Chen, 503-555-0180" — no intent words at all, routed correctly
from conversation history. Currently 18/18. The first run scored 15/17
and the two failures were both real: one prompt bug, one wrong label of
mine.

**Mixed intents: complaint always wins.** "I waited 40 minutes and I'm
furious, anyway can I book a cleaning Tuesday" routes to `complaint`
with `booking` recorded as secondary. The rule is asymmetric cost — a
booking the patient did not get, they ask for again; a complaint never
recorded is gone, and the clinic never learns of it. The first version
let the model weigh them and it chose booking.

**The secondary-intent note offers, it does not assert.** The reply says
"I haven't logged that as a complaint yet — say the word and I will",
not "I've also logged your complaint". Claiming work that was not done
is a lie the patient discovers when nobody calls back. A test enforces
the wording.

**A fourth intent, `other`, exists because real chat boxes receive
"hi".** Routing a greeting to the RAG agent produces "I don't have that
in the clinic's information" — technically true, terrible first
impression. `other` covers two different cases and answers them
differently: a pleasantry gets a greeting, while a genuinely out-of-scope
question gets told plainly it is out of scope.

**Classification failure defaults to `inquiry`.** It is the only agent
that refuses gracefully when it turns out to be wrong. Defaulting to
booking would interrogate a confused user; defaulting to complaint would
manufacture a record of something that never happened.

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
