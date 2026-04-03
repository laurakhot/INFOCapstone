"""
Event handlers for the laptop health alert bot.

Events handled:
  message.im        — user sends a DM to the bot (Q&A)
  app_home_opened   — user opens the App Home tab
"""

import home
from rag.articles import get_article, REPLACEMENT_METRICS
from alerts.templates import METRIC_LABELS


# In-memory store tracking the most recent alert per user.
# Maps user_id -> {"metric_key": str, "device": str}
# Used to provide contextual Q&A answers.
_recent_alerts = {}  # type: dict


def record_alert(user_id: str, metric_key: str, device: str) -> None:
    """Called by the alert sender after successfully sending an alert."""
    _recent_alerts[user_id] = {"metric_key": metric_key, "device": device}


def register_event_handlers(app) -> None:
    """Register all event handlers on the Bolt app."""

    # ── App Home opened ────────────────────────────────────────────────────
    @app.event("app_home_opened")
    def handle_app_home_opened(event, client):
        user_id = event["user"]
        client.views_publish(
            user_id=user_id,
            view=home.build_home_view(),
        )

    # ── Direct message to the bot ──────────────────────────────────────────
    @app.event("message")
    def handle_dm_message(event, client, say):
        # Only handle DMs (channel type "im"), ignore bot's own messages
        if event.get("channel_type") != "im":
            return
        if event.get("bot_id"):
            return

        user_id = event.get("user")
        text = (event.get("text") or "").lower().strip()

        if not text:
            return

        # Look up the most recent alert for context
        alert_ctx = _recent_alerts.get(user_id)

        response = _build_qa_response(text, alert_ctx)
        say(response)


def _build_qa_response(text: str, alert_ctx) -> str:
    """
    Generate a contextual reply based on the user's message and their last alert.

    Keyword matching strategy:
    - If the user's message references a known metric or general health terms,
      return the relevant KB article.
    - Fall back to a friendly generic response.
    """
    # Map common phrases to metric keys
    keyword_map = {
        "memory": "avg_memory_utilization",
        "ram": "avg_memory_utilization",
        "slow": "avg_boot_time",
        "startup": "avg_boot_time",
        "boot": "avg_boot_time",
        "performance": "max_memory_pressure",
        "pressure": "max_memory_pressure",
        "cpu": "cpu_count",
        "processor": "cpu_count",
        "battery": "battery_cycle",
        "charge": "battery_cycle",
    }

    # Try to match a keyword in the user's message
    matched_metric = None
    for keyword, metric_key in keyword_map.items():
        if keyword in text:
            matched_metric = metric_key
            break

    # If no keyword match, fall back to the metric from the most recent alert
    if matched_metric is None and alert_ctx:
        matched_metric = alert_ctx.get("metric_key")

    if matched_metric:
        label = METRIC_LABELS.get(matched_metric, matched_metric)
        article = get_article(matched_metric)
        intro = f"Here's what I know about *{label}*:\n\n"
        return intro + article

    # Generic fallback
    return (
        "Hey there! 👋 I'm here to help with your laptop health.\n\n"
        "You can ask me about things like:\n"
        "• *Memory usage* — how to free up RAM\n"
        "• *Startup speed* — how to speed up boot time\n"
        "• *System performance* — how to reduce memory pressure\n"
        "• *Battery health* — what to do if your battery is degrading\n\n"
        "What's on your mind?"
    )
