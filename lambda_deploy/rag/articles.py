"""
RAG article loader.

Routing is driven by rag/escalation_logic.md — edit that file to change metric
types and canvas URLs.

Articles live in rag/articles/{platform}_{metric_key}.md where platform is
inferred from the device name (mac or windows).
"""

import re
from pathlib import Path
from typing import Optional

# ARTICLES_DIR = Path(__file__).parent / "articles"
_ESCALATION_FILE = Path(__file__).parent / "escalation_logic.md"


def _load_escalation_table() -> dict:
    """
    Parse rag/escalation_logic.md and return a dict of:
      {metric_key: {"type": str, "canvas_url": Optional[str]}}
    """
    table: dict = {}

    for line in _ESCALATION_FILE.read_text(encoding="utf-8").splitlines():
        # Match 3-column table rows: | metric_key | type | canvas_url |
        m = re.match(
            r"^\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|", line
        )
        if not m:
            continue
        key, kind, canvas_url = m.group(1), m.group(2), m.group(3)
        if key in ("metric_key", "---", "-", "|-"):
            continue
        table[key] = {
            "type": kind,
            "canvas_url": None if canvas_url == "-" else canvas_url,
        }

    return table


# Load once at module import time
_TABLE = _load_escalation_table()


def _detect_platform(device: str) -> str:
    """Infer platform from device name. Defaults to mac."""
    d = device.lower()
    if "hp" in d or "windows" in d or "elitebook" in d or "thinkpad" in d:
        return "windows"
    return "mac"


# def get_article(metric_key: str, device: str = "") -> str:
#     """
#     Load the KB article for a given metric key and return it as Slack mrkdwn.
#     Pass device name so platform-specific articles (mac vs windows) are resolved.
#     Returns a fallback string if no article file is found.
#     """
#     plat = _detect_platform(device)
#     path = ARTICLES_DIR / f"{plat}_{metric_key}.md"

#     if not path.exists():
#         return (
#             f"*No self-help article found for this issue yet.*\n\n"
#             f"Please contact IT directly and mention: `{metric_key}`"
#         )
#     raw = path.read_text(encoding="utf-8")
#     return _md_to_mrkdwn(raw)


def get_canvas_url(metric_key: str) -> Optional[str]:
    """Return the Slack Canvas URL for a metric, or None if not configured."""
    entry = _TABLE.get(metric_key)
    if entry is None:
        return None
    return entry.get("canvas_url")


def is_replacement(metric_key: str) -> bool:
    """Return True if this metric requires hardware replacement (not self-serviceable)."""
    entry = _TABLE.get(metric_key)
    if entry is None:
        return False
    return entry.get("type") == "replacement"


def _md_to_mrkdwn(text: str) -> str:
    """
    Convert a subset of Markdown to Slack mrkdwn.

    Handles:
    - # / ## / ### headings  →  *Heading*
    - **bold**               →  *bold*
    - __bold__               →  *bold*
    - *italic* / _italic_    →  _italic_
    - - list items           →  • list items
    - Horizontal rules (---) are removed
    """
    lines = text.splitlines()
    output = []

    for line in lines:
        heading_match = re.match(r"^#{1,3}\s+(.*)", line)
        if heading_match:
            output.append(f"*{heading_match.group(1).strip()}*")
            continue

        if re.match(r"^-{3,}$", line.strip()):
            output.append("")
            continue

        line = re.sub(r"^(\s*)-\s+", r"\g<1>• ", line)
        line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)
        line = re.sub(r"__(.+?)__", r"*\1*", line)

        output.append(line)

    return "\n".join(output).strip()