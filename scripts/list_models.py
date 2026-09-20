"""List the models this API key can actually use.

Providers retire model names without warning, which shows up as a confusing
404 at runtime. Run this to see the live catalogue:

    python -m scripts.list_models
"""

from __future__ import annotations

import sys

from app.config import settings
from app.llm import LLMError, _get_client


def main() -> int:
    if not settings.llm_configured:
        print("GROQ_API_KEY is not set. Copy .env.example to .env and add a key.")
        return 1

    try:
        client = _get_client()
        models = sorted(m.id for m in client.models.list().data)
    except LLMError as exc:
        print(f"Could not reach the provider: {exc}")
        return 1
    except Exception as exc:  # network, auth, anything the SDK raises
        print(f"Could not list models: {type(exc).__name__}: {exc}")
        return 1

    print(f"{len(models)} model(s) available to this key:\n")
    for model_id in models:
        marks = []
        if model_id == settings.llm_model:
            marks.append("<- LLM_MODEL")
        if model_id == settings.router_model:
            marks.append("<- ROUTER_MODEL")
        print(f"  {model_id:42} {' '.join(marks)}")

    configured = {settings.llm_model, settings.router_model}
    missing = configured - set(models)
    if missing:
        print(f"\nWARNING: configured but not available: {', '.join(sorted(missing))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
