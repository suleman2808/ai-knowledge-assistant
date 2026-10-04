"""Phone number handling, shared by storage and lookup.

Its own module because both `analytics` (which writes numbers) and
`patients` (which reads them) need the same rules, and importing either
from the other would be circular.
"""

from __future__ import annotations

import re

# Phone-shaped text: enough digits, and the separators people type.
PHONE_IN_TEXT = re.compile(r"(?<![\w-])\+?\d[\d\s().-]{6,}\d(?![\w-])")

# Numbers are keyed on their last this-many digits, so 503-555-0180,
# (503) 555 0180 and +1 503 555 0180 are one person rather than three.
# Ten suits the US numbering of the demo data; another market would need
# a different rule, which is why this is named rather than buried in a
# slice.
KEY_DIGITS = 10
MIN_DIGITS = 7


def normalise(phone: str | None) -> str | None:
    """Reduce a phone number to a comparable key, or None if it isn't one."""
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) < MIN_DIGITS:
        return None
    return digits[-KEY_DIGITS:]


def phone_in(text: str | None) -> str | None:
    """Find the first phone number in a message and return its key."""
    for candidate in PHONE_IN_TEXT.findall(text or ""):
        key = normalise(candidate)
        if key:
            return key
    return None
