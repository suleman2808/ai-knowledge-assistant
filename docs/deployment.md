# Deployment

The project runs as a single container: FastAPI serves the API, the chat
UI and the dashboard, with the vector store baked into the image.

## Hugging Face Spaces (free)

Spaces is the right free host here because the app needs ~1 GB of RAM to
hold the embedding model. Render's free tier gives 512 MB and the
container is killed during startup; Spaces gives 16 GB on the free CPU
tier.

### 1. Create the Space

[huggingface.co/new-space](https://huggingface.co/new-space)

- **Space SDK:** Docker → *Blank*
- **Hardware:** CPU basic (free)
- **Visibility:** Public

### 2. Add the API key as a secret

In the Space: **Settings → Variables and secrets → New secret**

| Name | Value |
| --- | --- |
| `GROQ_API_KEY` | your key from console.groq.com |

Secrets are injected as environment variables, which is exactly what
`app/config.py` reads. Nothing in the repository changes, and the key is
never committed.

### 3. Push the code

```bash
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space main
```

### 4. The Space README

Spaces reads configuration from YAML front matter at the top of
`README.md`. Creating the Space through the web UI generates one; adding
this block to the top of the project README makes the repository
deployable directly:

```yaml
---
title: AI Knowledge Assistant
emoji: 🦷
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

GitHub renders that block as a small table, which is the cost of having
one repository serve both.

### What to expect

The first build takes 5–10 minutes: installing dependencies, then
running ingestion, which downloads the ONNX embedding model and builds
the vector store into the image. Later builds are faster because the
dependency layer is cached.

Free Spaces sleep after 48 hours of inactivity and wake on the next
request, which takes about 30 seconds. Worth knowing before sending a
client a link cold.

## Notes for any host

**Google Calendar will not work.** The OAuth flow is interactive and
`token.json` is deliberately not committed, so a deployed instance falls
back to the in-memory calendar. Bookings are accepted, clash-checked and
confirmed, but exist only in that process. For a real deployment, the
correct fix is a service account with domain-wide delegation, or storing
a refresh token as a secret — not shipping a token file.

**Analytics are ephemeral.** `data/analytics.db` lives in the container
filesystem, which most free hosts reset on restart. Spaces offers
persistent storage on paid tiers; Postgres or Turso would be the free
alternatives.

**`/api/analytics` is unauthenticated.** Acceptable for a demo with
invented data. Before deploying anything with real patient questions in
it, put that endpoint behind authentication.

**Set a rate limit you can afford.** The in-process limiter allows 30
requests per minute per client, which protects against a stuck loop but
not against a crowd. Groq's free tier has its own limits, and the app
degrades politely when they are hit.

## Running the container locally

```bash
docker build -t dental-assistant .
docker run --rm -p 7860:7860 -e GROQ_API_KEY=gsk_... dental-assistant
```

Then open http://127.0.0.1:7860.
