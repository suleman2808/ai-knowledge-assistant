# Deployment

The project runs as a single container: FastAPI serves the API, the chat
UI and the dashboard, with the vector store baked into the image.

## What it needs

Measured on a running instance, after loading the embedding model and
answering a question: **~290 MB resident**. Ingestion needs a little
more, and the image is around 700 MB.

That number is the ONNX switch paying off twice. On PyTorch the
footprint was roughly 1 GB, which ruled out every 512 MB free tier. At
290 MB the app fits comfortably on Hugging Face Spaces, Render, Fly.io
or a small VPS.

## Streamlit Community Cloud (free, no card)

The only host tried here that is free with **no card, no quota and no
sleep**. Render asks for a card even on its free tier, and a new Hugging
Face account gets `cpu-basic` quota of zero, which cannot be raised
without upgrading.

The trade-off: it runs Streamlit apps only, so the interface is
`streamlit_app.py` rather than the FastAPI app and the HTML UI. That
file is deliberately thin — every decision still happens in
`app.graph.run`, the same router, agents and retrieval. If the two
front-ends ever disagree about behaviour, that is a bug in the Streamlit
one.

1. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
2. **Create app** → **Deploy a public app from GitHub**
   - Repository: `suleman2808/ai-knowledge-assistant`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
3. **Advanced settings** → **Secrets**, paste:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

4. **Deploy**

First boot takes a few minutes: installing dependencies, then building
the vector store, since Streamlit Cloud has no build step. The result is
cached for the life of the container.

The app reads secrets through `st.secrets` and copies them into the
environment before `app.config` is imported, so nothing else in the
project needs to know which platform it is on.

## Cloudflare Tunnel (free, no account)

For showing the **real** FastAPI UI — and real Google Calendar bookings,
which no deployment can do, because the OAuth token is deliberately not
committed.

```bash
# one terminal
python -m app.main

# another terminal
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints a public `https://<random>.trycloudflare.com` URL
that works from anywhere while the command runs. No account, no card, no
configuration. Download it from
[Cloudflare's releases](https://github.com/cloudflare/cloudflared/releases)
— a single executable.

The link dies when the command stops, so this is for a live call rather
than a CV.

## Render — needs a card

Render's free tier still requires a card on file (a $1 authorisation,
not a charge). If that is acceptable, this is the smoothest route,
because it authenticates with the GitHub account the repository already
lives in.

1. [render.com](https://render.com) → **Get Started** → sign in with GitHub
2. **New +** → **Blueprint**, and select this repository — `render.yaml`
   configures the service
3. When prompted for `GROQ_API_KEY`, paste the key from
   console.groq.com. Render stores it as a secret.
4. **Apply**

Or without the blueprint: **New +** → **Web Service** → pick the repo →
runtime **Docker**, instance type **Free**, then add `GROQ_API_KEY` under
Environment.

Render injects a `PORT` variable and kills any container that does not
listen on it — the message is "no open ports detected", which does not
say why. `docker-entrypoint.sh` prefers `PORT` and falls back to
`APP_PORT`, so the same image serves both Render and Spaces.

Free services sleep after 15 minutes of inactivity and take about 50
seconds to wake. Open the link yourself before sending it to anyone.

## Hugging Face Spaces (free)

The easiest free option: 16 GB on the free CPU tier, Docker support, and
secrets that arrive as environment variables.

**A new account may get no free quota at all.** Verifying the email is
not always enough. The API reports
`Quota exceeded for flavor cpu-basic (requested=1): current=0, limit=0`,
and the Space settings page says "You've reached your cpu-basic quota
limit, please upgrade your account, or pause your previous Spaces" —
even with a single Space and nothing else running. `limit=0` is the
tell: no entitlement at all. It is an account problem wearing the
costume of a capacity problem, and nothing in the repository can fix
it.

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

## Other hosts

Any of these work at ~290 MB:

| Host | Free tier | Notes |
| --- | --- | --- |
| **Render** | 512 MB | Sleeps after 15 min idle; ~50s cold start |
| **Fly.io** | 256 MB shared | Tight — use a 512 MB machine |
| **Railway** | trial credit | No sleeping while credit lasts |
| **Any VPS** | — | 1 GB is plenty |

All read `GROQ_API_KEY` from the environment and need no code change.

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
