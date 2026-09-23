#!/bin/sh
# Start the server on whichever port the host asked for.
#
# Hosts disagree about this. Hugging Face Spaces expects a fixed 7860.
# Render, Railway, Fly and Cloud Run inject a PORT environment variable
# and expect the app to honour it — a container that ignores PORT is
# marked unhealthy and killed, with a "no open ports detected" message
# that does not say why.
#
# So: use PORT when the host sets one, otherwise fall back to APP_PORT,
# otherwise 7860. One image that runs anywhere.
exec python -m uvicorn app.main:app \
    --host "${APP_HOST:-0.0.0.0}" \
    --port "${PORT:-${APP_PORT:-7860}}"
