"""
RAG article loader.

Articles are stored as .md files in rag/articles/.
File names must exactly match the metric key (e.g. avg_memory_utilization.md).

Add your own support / KB articles to that folder — no code changes needed.
"""

import re
from pathlib import Path

ARTICLES_DIR = Path(__file__).parent / "articles"

# Metrics that require hardware replacement (user cannot self-fix)
REPLACEMENT_METRICS = {"cpu_count", "battery_cycle"}


def get_article(metric_key: str) -> str:
    """
    Load the KB article for a given metric key and return it as Slack mrkdwn.
    Returns a fallback string if no article file is found.
    """
    path = ARTICLES_DIR / f"{metric_key}.md"
    if not path.exists():
        return (
            f"*No self-help article found for this issue yet.*\n\n"
            f"Please contact IT directly and mention: `{metric_key}`"
        )
    raw = path.read_text(encoding="utf-8")
    return _md_to_mrkdwn(raw)


def is_replacement(metric_key: str) -> bool:
    """Return True if this root cause requires hardware replacement."""
    return metric_key in REPLACEMENT_METRICS


def _md_to_mrkdwn(text: str) -> str:
    """
    Convert a subset of Markdown to Slack mrkdwn.

    Handles:
    - # / ## / ### headings  →  *Heading*
    - **bold**               →  *bold*
    - __bold__               →  *bold*
    - *italic* / _italic_    →  _italic_
    - Numbered and bullet lists are kept as-is (Slack renders them)
    - Horizontal rules (---) are removed
    """
    lines = text.splitlines()
    output = []

    for line in lines:
        # Headings: # H1, ## H2, ### H3
        heading_match = re.match(r"^#{1,3}\s+(.*)", line)
        if heading_match:
            output.append(f"*{heading_match.group(1).strip()}*")
            continue

        # Horizontal rules
        if re.match(r"^-{3,}$", line.strip()):
            output.append("")
            continue

        # Bold: **text** or __text__
        line = re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)
        line = re.sub(r"__(.+?)__", r"*\1*", line)

        # Italic: *text* or _text_ (only single markers, not already bold)
        # Use negative lookbehind/ahead to avoid touching already-converted *bold*
        line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", line)

        output.append(line)

    return "\n".join(output).strip()
