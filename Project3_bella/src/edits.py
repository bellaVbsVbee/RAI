"""Two helpers for editing a Findings text (Week 2). The edits themselves are yours to write."""

import re

_SENTENCE_END = re.compile(r"(?<=[.])\s+")


def sentences(text: str) -> list[str]:
    """Split a report into sentences (on a full stop followed by a space)."""
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def find_sentence(text: str, word: str = "effusion") -> int | None:
    """Index of the first sentence that contains `word` (case-insensitive), or None if there is none."""
    for i, s in enumerate(sentences(text)):
        if re.search(word, s, re.IGNORECASE):
            return i
    return None
