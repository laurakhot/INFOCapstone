"""
RAG article loader.

Routing is driven by rag/escalation_logic.md — edit that file to change which
article each metric maps to, and whether it requires IT replacement or not.

Articles live in:
  rag/articles/{article}.md                        — shared (platform=shared)
  rag/articles/mac/{article}_mac.md                — Mac-specific
  rag/articles/windows/{article}_windows.md        — Windows/HP-specific

Platform is inferred from the device name passed to get_article().
"""

import re
from pathlib import Path
from typing import Optional

ARTICLES_DIR = Path(__file__).parent / "articles"
_ESCALATION_FILE = Path(__file__).parent / "escalation_logic.md"


def _load_escalation_table() -> tuple[dict, Optional[dict]]:
    """
    Parse rag/escalation_logic.md and return:
      - table: dict of {metric_key: {"article": str, "type": str, "platform": str}}
      - wildcard: fallback entry for unspecified metrics, or None
    """
    table: dict = {}
    wildcard: Optional[dict] = None

    for line in _ESCALATION_FILE.read_text(encoding="utf-8").splitlines():
        # Match 4-column table rows: | key | article | type | platform |
        m = re.match(
            r"^\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|", line
        )
        if not m:
            continue
        key, article, kind, platform = m.group(1), m.group(2), m.group(3), m.group(4)
        # Skip header and separator rows
        if key in ("metric_key", "---", "-", "|-"):
            continue
        entry = {"article": article, "type": kind, "platform": platform}
        if key == "*":
            wildcard = entry
        else:
            table[key] = entry

    return table, wildcard


# Load once at module import time
_TABLE, _WILDCARD = _load_escalation_table()


def _get_routing_entry(metric_key: str) -> dict:
    """Return the routing entry for metric_key, falling back to the wildcard."""
    if metric_key in _TABLE:
        return _TABLE[metric_key]
    if _WILDCARD is not None:
        return _WILDCARD
    return {"article": "{metric_key}", "type": "self-service", "platform": "shared"}


def _detect_platform(device: str) -> str:
    """Infer platform from device name. Defaults to mac."""
    d = device.lower()
    if "hp" in d or "windows" in d:
        return "windows"
    return "mac"


def get_article(metric_key: str, device: str = "") -> str:
    """
    Load the KB article for a given metric key and return it as Slack mrkdwn.
    Pass device name so platform-specific articles (mac vs windows) are resolved.
    Returns a fallback string if no article file is found.
    """
    entry = _get_routing_entry(metric_key)
    article_name = entry["article"]
    platform_flag = entry.get("platform", "shared")

    if article_name == "{metric_key}":
        article_name = metric_key

    if platform_flag == "shared":
        path = ARTICLES_DIR / f"{article_name}.md"
    else:
        plat = _detect_platform(device)
        path = ARTICLES_DIR / plat / f"{metric_key}.md"

    if not path.exists():
        return (
            f"*No self-help article found for this issue yet.*\n\n"
            f"Please contact IT directly and mention: `{metric_key}`"
        )
    raw = path.read_text(encoding="utf-8")
    return _md_to_mrkdwn(raw)


def is_replacement(metric_key: str) -> bool:
    """Return True if this root cause requires hardware replacement."""
    return _get_routing_entry(metric_key).get("type") == "replacement"


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
        # Headings: # H1, ## H2, ### H3
        heading_match = re.match(r"^#{1,3}\s+(.*)", line)
        if heading_match:
            output.append(f"*{heading_match.group(1).strip()}*")
            continue

        # Horizontal rules
        if re.match(r"^-{3,}$", line.strip()):
            output.append("")
            continue

        # Unordered list items: - item → • item
        line = re.sub(r"^(\s*)-\s+", r"\g<1>• ", line)

        # Italic first: *text* (single markers only — lookahead/behind excludes **bold**)
        line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", line)

        # Bold: **text** or __text__ (runs after italic so *bold* isn't re-matched)
        line = re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)
        line = re.sub(r"__(.+?)__", r"*\1*", line)

        output.append(line)

    return "\n".join(output).strip()
