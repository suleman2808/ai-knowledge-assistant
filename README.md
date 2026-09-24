---
title: AI Knowledge Assistant
emoji: 🦷
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AI Knowledge Assistant — RAG + Multi-Agent Router

**▶ [Try it live](https://ai-knowledge-assistant-original.streamlit.app/)**

[![tests](https://github.com/suleman2808/ai-knowledge-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/suleman2808/ai-knowledge-assistant/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![demo](https://img.shields.io/badge/demo-streamlit-0f5c63)](https://ai-knowledge-assistant-original.streamlit.app/)

Ask it something the clinic's documents cover ("how much is a root canal
on a molar?"), and something they do not ("do you offer botox?"). It
answers the first with a citation and declines the second — that
difference is the whole point of the project.

A customer-facing assistant for a service business, built around one
constraint: **it must not make things up about the business.** The demo
domain is a dental clinic.

An incoming message is classified by a router and dispatched to one of
three specialist agents:

| Agent | Responsibility |
| --- | --- |
| **Inquiry** | Answers questions about services, hours, pricing and policies using RAG over the clinic's own documents. Never answers from model knowledge, and refuses when the documents do not cover the question. |
| **Booking** | Checks availability and creates real appointments in Google Calendar. |
| **Complaint** | Rates severity, logs the complaint, and escalates the serious ones by rule rather than by model judgement. |

Every turn is recorded, and the analytics dashboard reports what patients
asked, how often the assistant could answer, and which questions it could
not — the gaps in the clinic's own documentation.

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

The diagram above is hand-drawn for readability; the authoritative one is
generated from the compiled graph in
[docs/architecture.md](docs/architecture.md), regenerated with
`python -m scripts.draw_graph --write`, so it cannot drift from the code.

## Quickstart

Requires **Python 3.11+** (developed on 3.13) and a free
[Groq API key](https://console.groq.com) — no card, no paid services
anywhere in this project.

```bash
git clone https://github.com/suleman2808/ai-knowledge-assistant.git
cd ai-knowledge-assistant

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
#   then put your Groq key in .env

python -m scripts.smoke_test    # checks config and the provider
python -m scripts.ingest        # builds the knowledge base (~30s first run)
python -m app.main              # http://127.0.0.1:8000
```

| | |
| --- | --- |
| `http://127.0.0.1:8000/` | chat UI |
| `http://127.0.0.1:8000/analytics` | activity dashboard |
| `http://127.0.0.1:8000/docs` | interactive API documentation |

Google Calendar is **optional** — see below. Without it the Booking Agent
uses an in-memory calendar that enforces real clash detection, so a fresh
clone works end to end with nothing but a Groq key.

## Stack

| | |
| --- | --- |
| **LangGraph** | Orchestration as an explicit, inspectable graph |
| **Groq** | `gpt-oss-120b` for agents, `gpt-oss-20b` for routing — free tier |
| **all-MiniLM-L6-v2 on ONNX Runtime** | Embeddings computed locally: no API, no key, no rate limit |
| **ChromaDB** | Persisted local vector store |
| **BM25** | Keyword search, implemented directly (~60 lines) |
| **FastAPI** | HTTP layer with auto-generated interactive docs |
| **SQLite** | Analytics and complaint records |
| **Google Calendar API** | Real bookings, with an in-memory fallback |
| **Plain HTML/JS** | No framework, no build step, no CDN |

## Command-line tools

Every layer can be exercised on its own, without the ones above it.

```bash
# Knowledge base
python -m scripts.inspect_chunks --show 2      # chunking, without embedding
python -m scripts.search "how much is a root canal"
python -m scripts.search --calibrate           # relevance threshold evidence
python -m scripts.eval_retrieval --compare     # vector vs hybrid, measured

# Agents, individually
python -m scripts.try_agent inquiry --suite
python -m scripts.try_agent booking "book a cleaning next Tuesday at 2pm"
python -m scripts.try_agent complaint "I was charged twice"

# The assembled graph
python -m scripts.chat                         # interactive, keeps history
python -m scripts.chat --routing               # routing accuracy probe
python -m scripts.chat --scenario booking      # scripted multi-turn booking

# Analytics
python -m scripts.seed_demo --reset            # realistic traffic, real graph
python -m scripts.analytics_report

# Diagnostics
python -m scripts.list_models                  # what your key can reach
python -m scripts.google_auth --check
python -m scripts.verify_fresh_clone           # does it work for someone else?
```

## Project layout

```
app/
  config.py          Typed configuration, loaded once from .env
  llm.py             The only module that talks to an LLM provider
  schemas.py         Request/response models
  main.py            FastAPI: chat, streaming, health, analytics, UI
  agents/            The three specialists, each a plain function
  prompts/           Prompt templates as .md files
  graph/             LangGraph state, router and wiring
  rag/               Chunking, embeddings, BM25, vector store, retrieval
  integrations/      Calendar, analytics
streamlit_app.py     Alternative front-end, for free hosting
data/documents/      The clinic's documents — the knowledge base
ui/                  Chat page and dashboard
scripts/             CLI entry points
tests/               174 tests
```

## Optional: Google Calendar

Skip this and everything still works.

1. [console.cloud.google.com](https://console.cloud.google.com/) → new project
2. APIs & Services → Library → **Google Calendar API** → Enable
3. Google Auth Platform → **Audience** → add your own address as a test user
4. Credentials → Create Credentials → OAuth client ID → **Desktop app**
5. Download the JSON as `credentials.json` in the project root (gitignored)

```bash
python -m scripts.google_auth          # authorise once, in a browser
python -m scripts.google_auth --check  # verify, list free slots
python -m scripts.google_auth --revoke # back to the in-memory calendar
```

## Deployment

```bash
docker build -t dental-assistant .
docker run --rm -p 7860:7860 -e GROQ_API_KEY=gsk_... dental-assistant
```

The image builds the vector store at build time, so the container starts
ready to answer, and it runs on ~290 MB — which fits every free tier.
`render.yaml` makes Render a one-click import; Hugging Face Spaces is
configured by the front matter at the top of this file. The same image
serves both: `docker-entrypoint.sh` uses `PORT` where the host injects
one and a fixed port where it does not.

**Free hosting with no card.** Render and Hugging Face both turned out
to want a payment method or an account upgrade. `streamlit_app.py` is a
thin alternative front-end for Streamlit Community Cloud, which is free
with no card and no quota — it calls the same `app.graph.run`, so the
router, agents and retrieval are identical and there is no second copy
of the logic. For demonstrating the *real* FastAPI UI, including live
Google Calendar bookings, a Cloudflare quick tunnel exposes the local
app with no account at all.

See [docs/deployment.md](docs/deployment.md) for all of these, and for
what does *not* survive a deployment — Google Calendar (the OAuth token
is deliberately not committed) and the analytics database (ephemeral on
free hosts).

## Optional: the PyTorch embedding runtime

The default runtime is ONNX, which needs no PyTorch. To use a different
Hugging Face embedding model:

```bash
pip install -r requirements-torch.txt
# set EMBEDDING_BACKEND=sentence-transformers in .env
python -m scripts.ingest
```

## Tests

```bash
python -m pytest tests/ -q      # 174 tests, ~13s
```

Every LLM call, vector store and calendar is stubbed, so the suite needs
no key, no network and no ingested index. What is tested is the logic
around the model — grounding enforcement, escalation rules, threshold
behaviour, failure handling — rather than the model itself.

The whole thing is also verified from a genuine fresh clone:

```bash
python -m scripts.verify_fresh_clone
```

A new checkout into a temporary directory, a new virtualenv,
`requirements.txt` only, then the README's quickstart in order — smoke
test, ingest, tests, CLI, server, and an HTTP request whose answer must
come back grounded and cited. The clone is also checked for leaked
secrets. The only thing carried across is the `.env` key, because that
cannot be invented.

This catches the failures that only happen to somebody else: a file that
exists on the developer's machine but was never committed, a dependency
installed by hand months ago, a setup step that lives only in somebody's
memory. Running the app in the directory you built it in proves none of
that.

---

# Design notes

The decisions worth defending, grouped by what they are about. Several
describe bugs found during development; those are kept because how a
system fails is more informative than a list of features.

## Not making things up

**A two-stage defence, because one stage provably is not enough.**
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

No single threshold separates them: the cinema question outscores two
legitimate ones and retrieves the clinic's opening-hours table, which a
naive agent would answer from. So:

1. The similarity threshold (0.30) rejects the obviously unrelated.
2. The Inquiry Agent's prompt requires that the retrieved context answers
   *this specific question*, and emits a refusal token when it does not.

Stage two is what catches the cinema case, verified end to end.

**"I don't know" must not mean "the answer depends".** "What's the
cancellation fee if I cancel tomorrow?" retrieved the right section
first, and the model refused anyway, because "tomorrow" does not say how
many hours' notice that is. The grounding rule now distinguishes a
question the material cannot answer from one whose answer depends on a
detail the patient omitted — the latter gets the rule for each case.
Refusing there withholds an answer the clinic has written down, which is
the opposite of what the rule is for.

**Every answer shows its provenance.** The UI badges each reply as
*From clinic documents* or *Not in our documents*, with the cited
sections expandable underneath. Grounding is the promise this project
makes, so it is visible to the patient rather than buried in a developer
console.

**How retrieval failures are handled.** `retrieve()` never raises for an
expected condition. It returns a `RetrievalResult` carrying a status the
agent branches on: `OK`, `LOW_CONFIDENCE` (matches exist but all score
below the threshold), `NO_MATCH`, `EMPTY_INDEX` (nothing ingested — an
operator error, distinct from a user one) and `UNAVAILABLE` (the store or
model failed). Exceptions are reserved for genuine bugs, so no ordinary
production condition can produce a stack trace in front of a customer.

## Retrieval

**Chunking splits on document structure, not character count**, so each
chunk is a unit the author already decided was coherent. Three
consequences:

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

**Hybrid search: vector plus BM25.** Vector search matches by meaning,
which is its strength and its blind spot. "my tooth got knocked out"
finds a section titled "First Aid" with no shared words; "do you take
cigna" scored 0.23 against the page that literally lists *Cigna Dental
PPO*, because an embedding model gives a brand name little semantic
weight. BM25 is the complement — it scores exact term overlap and weights
rare terms heavily. The two rankings are combined by Reciprocal Rank
Fusion, which works on ranks rather than raw scores, because cosine
similarity and BM25 scores live on unrelated scales and cannot
meaningfully be added. BM25 is implemented directly rather than pulled in
as a dependency, because the algorithm is short and worth being able to
read.

Measured on 28 labelled questions (`scripts/eval_retrieval.py`):

| | vector only | hybrid |
| --- | --- | --- |
| hit@1 | 68% | 79% |
| hit@k | 79% | 93% |
| MRR | 0.720 | 0.848 |
| out-of-scope admitted | 3/8 | 3/8 |

**How a keyword hit earns the right to bypass the threshold.** Three
conditions: top-two BM25 rank, a matched term appearing in at most three
chunks, and IDF coverage of at least 0.575 — the share of the query's
information that the chunk actually matched.

The coverage condition exists because the first version did not have it,
and doubled out-of-scope false positives from 3/8 to 6/8: "renew my
**car** insurance" matched *"surface **car** park"*, "**reset** my email
password" matched *"benefits **reset** on 1 January"*. A rare word that
merely appears in a question is not the same as a rare word that *is* the
question. Vector score does not separate those cases — the correct aetna
hit scores 0.08 while the wrong cinema hit scores 0.23 — but IDF coverage
does: legitimate hits measured 0.63–1.00, coincidental ones 0.21–0.52.
The cut-off is the midpoint of that gap, fitted on 14 examples, so it is
calibrated rather than proven.

A known limitation, covered by a test: query words absent from every
document get maximal IDF, which is exactly what correctly sinks
"**renew** my car insurance", but also means unfamiliar slang lowers
coverage. It fails safe — back to vector search, at worst a refusal,
never a false answer.

**Follow-up questions are rewritten before retrieval.** "Is that for one
surface?" retrieves nothing useful, because the word carrying the meaning
— "filling" — is in the previous turn. The Inquiry Agent rewrites
follow-ups into standalone questions using the small model, only when
history exists, so a first message costs no extra call. It falls back to
the original query on any failure, and rejects a rewrite that is
suspiciously long, which is what a model answering instead of rewriting
produces.

**Why cosine distance is set explicitly.** Chroma defaults to squared L2.
With normalised vectors the ranking would be the same, but the distance
range differs, which would make a fixed relevance threshold meaningless.
The threshold is what lets the Inquiry Agent say "I don't know" instead
of answering from a weak match.

## Orchestration and routing

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
inquiry, "I need to cancel Tuesday" is a booking), an angry question that
is still a question, and a message whose only content is "Sarah Chen,
503-555-0180" — no intent words at all, routed correctly from
conversation history. Currently 18/18. The first run scored 15/17 and
both failures were real: one prompt bug, one wrong label of mine.

**Mixed intents: complaint always wins.** "I waited 40 minutes and I'm
furious, anyway can I book a cleaning Tuesday" routes to `complaint` with
`booking` recorded as secondary. The rule is asymmetric cost — a booking
the patient did not get, they ask for again; a complaint never recorded
is gone, and the clinic never learns of it. The first version let the
model weigh them and it chose booking.

**The secondary-intent note offers, it does not assert.** The reply says
"I haven't logged that as a complaint yet — say the word and I will", not
"I've also logged your complaint". Claiming work that was not done is a
lie the patient discovers when nobody calls back. A test enforces the
wording.

**A fourth intent, `other`, exists because real chat boxes receive
"hi".** Routing a greeting to the RAG agent produces "I don't have that
in the clinic's information" — technically true, terrible first
impression. `other` covers two different cases and answers them
differently: a pleasantry gets a greeting, while a genuinely out-of-scope
question is told plainly that it is out of scope.

**Classification failure defaults to `inquiry`.** It is the only agent
that refuses gracefully when it turns out to be wrong. Defaulting to
booking would interrogate a confused user; defaulting to complaint would
manufacture a record of something that never happened.

## The agents

**Agents are plain functions, not graph nodes.** Each takes a message and
returns an `AgentResponse`. None imports LangGraph, FastAPI or another
agent, so each is testable alone, and the graph was wired up later
without changing a line of agent code.

**Prompts are markdown files, not string literals.** The person who
should approve what a clinic says to patients does not read Python.
Editing a prompt produces a readable diff, and tuning becomes a content
change rather than a code change. Placeholders use `{{name}}` because
prompts contain literal JSON braces that `str.format` would choke on.

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
call to both judge severity and write an apology biases the judgement — a
model composing a soothing response mirrors the patient's tone, so an
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

## Calendar

**Timezones are converted at exactly one boundary.** The rest of the
project uses naive datetimes meaning clinic wall-clock time, which is the
right model for a business whose hours are "Monday 8am to 5pm" regardless
of where the patient is sitting. Google works in absolute time.
`app/integrations/google_calendar.py` is the only module that converts,
and it is tested across a DST boundary: 2pm is `-08:00` in January and
`-07:00` in July. An appointment written an hour out is worse than no
appointment, because the patient turns up.

**The OAuth consent flow lives in a script, not in the app.** It opens a
browser, which must never happen inside an HTTP request — the server
would block on a dialogue nobody can see. At runtime the backend only
*loads* a stored token, refreshing it silently when expired; if there is
no usable token it raises, and `get_calendar()` falls back to the
in-memory calendar.

**The scope requested is `calendar.events`, not `calendar`,** so the
consent screen does not ask for permission to delete entire calendars.
That decision was nearly undone by accident: the health check called
`calendars.get`, which needs the broader scope, and failed with a 403
after a successful authorisation. Widening the scope would have taken
thirty seconds and quietly discarded the security decision; the check was
narrowed instead.

**A clash is re-checked immediately before writing.** Google has no
conditional insert, so a check cannot be atomic with the write. A
conversation takes seconds; re-checking narrows the race to milliseconds.
The remaining window is handled rather than ignored — the Booking Agent
treats a write-time clash as "offer alternatives", not as an error.

**Events the organiser marked free, and cancelled events, do not block
the diary**, and recurring events are expanded with `singleEvents` —
without it a weekly staff meeting blocks only its first week and every
other Tuesday looks bookable.

## Analytics

**Contact details are redacted before anything reaches disk.** Analytics
needs *what* patients asked, not *who* asked. Phone numbers and emails
are stripped at the single write boundary, so no caller can forget.
Probing the redactor found it also mangled ISO dates and turned the
complaint reference `CMP-20260920-4F2A` into `CMP-[phone]-4F2A` —
destroying the one identifier staff trace a complaint by. Both are now
covered by tests.

**Logging never breaks a conversation — except where it must.** A failed
turn write is logged and swallowed; an analytics outage is an
inconvenience, a chat that errors because of one is a defect. Complaints
are the exception: a failed complaint write is caught, held in memory,
logged at error level and flagged on the response, because a silently
lost complaint is the one failure this project treats as unacceptable.

**Logging lives in `run()`, not inside a graph node.** The graph stays
free of side effects beyond the agents' own, so it can be invoked in
tests or evaluations without polluting real numbers. The routing probe
passes `log=False`, and every test gets its own throwaway database via an
autouse fixture — before that fixture existed, the graph tests would have
written into the real analytics file.

**The first report found three retrieval bugs.** Of eight "knowledge
gaps" in the seeded data, only four were real. The other four had three
distinct causes: a bare brand name scoring below threshold, a correct
retrieval refused by an over-strict grounding check, and follow-up
questions retrieved without the conversation that gives them meaning. All
three are fixed above; the answer rate went from 62% to 81%. Analytics
doing its job on the system that produces it.

## API and interface

**Conversation history lives on the server.** Clients send only the new
message and a session id. A client-supplied transcript could be forged to
make the assistant believe it had already said something it had not — and
since follow-up rewriting reads that history, a forged transcript would
steer retrieval too. Sessions are in-memory with a TTL, which is also a
privacy decision: transcripts contain names, symptoms and complaints, and
keeping them on disk without a retention policy creates a problem the
project does not need.

**Progress is streamed, because tokens cannot be.** The agents' model
calls are not streamed, so there is nothing token-by-token to send. What
the graph can report is which node is running — "working out what you
need", then "searching the clinic's documents". A five-second wait with
visible progress reads as work; the same wait with a blank screen reads
as broken. Server-sent events carry those stages, then the finished
answer. Once a stream has started the status code is already 200, so a
failure cannot be reported as a 500 — it is delivered as an `error` event
inside the stream, and a test covers that.

**The UI has no framework, no build step and no CDN.** Roughly 300 lines
of plain JavaScript, served by the same FastAPI process. That includes a
small markdown renderer for the four things the model actually emits —
bold, italics, inline code and bullet lists — because a library is ~40 KB
for that. Model output is untrusted input: it derives from documents and
from whatever a patient typed, so it is escaped before any of our own
markup is added.

**Requests are rate limited** per client, because an unauthenticated
endpoint that spends money on model calls is otherwise an open tab. A
global exception handler guarantees no stack trace reaches a patient, and
the embedding model is warmed at startup so the first request does not
look broken.

## Infrastructure

**Why a single LLM module.** Nothing outside `app/llm.py` imports a
provider SDK. Switching from Groq to OpenAI, Anthropic or a local model
means rewriting one function. Retry policy, backoff and error translation
live there too, so failure handling is consistent rather than duplicated
at every call site. This was tested for real when the originally
specified Llama models were retired from Groq's catalogue mid-build: the
fix was twelve lines in one file.

**Why two models.** Intent classification into three buckets does not
need the large model. The router uses `gpt-oss-20b` at low reasoning
effort — faster and lighter on the free tier's rate limit; the agents use
`gpt-oss-120b`, where answer quality actually matters.

**Reasoning-model token budgets.** The `gpt-oss` models spend hidden
reasoning tokens from the same `max_tokens` budget as the visible reply.
Set the cap too low and the API returns success with an *empty* string
rather than an error. `app/llm.py` detects that case and raises an
explanatory error naming the real cause, because an empty reply is
otherwise extremely hard to diagnose.

**Why local embeddings.** Embedding a few hundred chunks through an API
means a key, a bill and a rate limit. `all-MiniLM-L6-v2` runs on CPU in
seconds and makes the retrieval half of the system work offline.

**Why ONNX rather than PyTorch.** The project began on
`sentence-transformers`. Mid-build, Windows Smart App Control started
refusing PyTorch's unsigned `c10.dll` (`WinError 4551`), and every
inquiry failed. The system degraded as designed — patients were told
search was unavailable rather than shown a stack trace — but "disable an
OS security feature" is not an acceptable install step, and anyone
cloning onto a locked-down Windows machine would hit the same wall.

The fix changed the runtime, not the model: Chroma ships
`all-MiniLM-L6-v2` as an ONNX export on Microsoft's signed `onnxruntime`.
Before switching, the new vectors were compared against the stored
PyTorch ones for all 74 chunks — cosine similarity 1.000000 throughout —
so the retrieval threshold calibrated earlier remained valid. Side
effects, all good: PyTorch (~1.5 GB) is no longer a required dependency,
and ingestion dropped from minutes to 8 seconds.

---

# Limitations

Stated rather than hidden. Each is a deliberate stopping point, not an
oversight.

**Security and privacy**

- `/api/analytics` has no authentication. Fine for a local demo; a real
  deployment needs auth, since the dashboard exposes patient questions.
- Names are not redacted from stored messages. Reliable detection needs
  an NER model; phone numbers and emails are handled.
- The rate limiter is in-process, so it protects a single instance.
  Behind a load balancer this belongs in the proxy or a shared store.

**Retrieval**

- The IDF coverage cut-off (0.575) is fitted on 14 examples. Calibrated,
  not proven; it should be re-derived on real traffic.
- Two known misses remain: "what should I avoid after having a tooth
  out" never uses the word *extraction*, a vocabulary gap neither method
  bridges; and "which bus goes to the clinic" was a hybrid win given up
  when the coverage rule tightened.
- **Query expansion was tried for the first of those and removed.** A
  curated map of patient phrasing to clinical terms ("tooth out" →
  "extraction") fed into the keyword search changed the evaluation by
  nothing at all: identical hit@1, hit@k and MRR. The reason is
  instructive. `extraction` has an IDF of only 2.45 in this corpus —
  it appears in several chunks, so it is not the distinctive term it
  looks like — while the generic words in the query (`after`, `tooth`,
  `out`, `avoid`) match every aftercare section equally. The right chunk
  is also long, so BM25's length normalisation pushes it further down.
  It never reached the top five. The map was deleted rather than kept
  for the look of it.
- No cross-encoder reranker. It is the standard next improvement and,
  unlike expansion, it would address the actual cause: these misses need
  a model that reads query and chunk *together*, not a better bag of
  words.

**Operational**

- Sessions and the in-memory calendar are lost on restart by design.
- Groq's free tier has variable latency — typically 1–2s, occasionally
  10s+. The streamed progress hides most of it.
- CI runs the suite on 3.11, 3.12 and 3.13, but not the fresh-clone
  check, which needs an API key.

# A note on the demo data

The clinic, its staff, prices and policies are fictional. The documents
in `data/documents/` were written to be realistic enough that chunking
and retrieval face real problems — pricing tables, nested policy
sections, clinical instructions, an FAQ — rather than to describe any
real practice. Phone numbers use the reserved `555-01xx` range.
