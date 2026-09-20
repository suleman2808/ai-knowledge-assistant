"""Load prompt templates from disk.

Prompts live in `.md` files beside this module rather than as string
literals in the agent code, for three reasons:

- They are the part of an LLM system most often edited, and editing a
  markdown file produces a readable diff rather than a wall of changed
  indentation inside a Python triple-quoted string.
- They can be reviewed by someone who does not read Python. For a clinic,
  the person who should sign off on what the assistant says to patients is
  not the person who wrote the agent.
- Tuning a prompt becomes a content change, not a code change.

Placeholders use `{{name}}` rather than Python's `{name}`, because prompts
frequently contain literal JSON braces and `str.format` would choke on
them.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent

PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class PromptError(RuntimeError):
    """A prompt file is missing, or a placeholder was left unfilled."""


@lru_cache(maxsize=32)
def load(name: str) -> str:
    """Read a prompt template by name, without its `.md` extension.

    Cached, since prompts do not change while the process runs.
    """
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in PROMPTS_DIR.glob("*.md")))
        raise PromptError(f"No prompt named {name!r}. Available: {available}")
    return path.read_text(encoding="utf-8").strip()


def render(name: str, **values: object) -> str:
    """Load a template and substitute its `{{placeholders}}`.

    Raises:
        PromptError: The template contains a placeholder with no value.
            Failing loudly matters here — a prompt silently sent with the
            literal text `{{context}}` in it produces a plausible-looking
            answer grounded in nothing at all.
    """
    template = load(name)
    rendered = PLACEHOLDER_RE.sub(
        lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0),
        template,
    )

    leftover = PLACEHOLDER_RE.findall(rendered)
    if leftover:
        raise PromptError(
            f"Prompt {name!r} has unfilled placeholder(s): {', '.join(sorted(set(leftover)))}"
        )
    return rendered
