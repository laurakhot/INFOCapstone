"""
Button action handlers for the laptop health alert bot.

Action IDs handled here:
  root_cause__{metric_key}           — user clicks a root cause button
  done_selfservice__{metric_key}     — user says they've fixed a self-serviceable issue
  done_replacement__{metric_key}     — user acknowledges a replacement-bound issue
  snooze                             — user snoozes the alert
  opt_out                            — user wants to stop receiving alerts
  opt_out_reason__{reason}           — user provides optional opt-out feedback
"""

import json
import os
import re

from alerts.templates import (
    build_article_blocks,
    build_acknowledged_blocks,
    build_snoozed_blocks,
    build_opted_out_blocks,
    build_opt_out_feedback_blocks,
    METRIC_LABELS,
)
from backend_client import run_diagnosis
from rag.articles import get_article, is_replacement

OPT_OUTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "opt_outs.json")


def _load_opt_outs() -> dict:
    try:
        with open(OPT_OUTS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_opt_outs(data: dict) -> None:
    with open(OPT_OUTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _get_username(body: dict) -> str:
    """Extract the acting user's Slack username from the action body."""
    return body.get("user", {}).get("username", "unknown")


def _get_user_id(body: dict) -> str:
    return body.get("user", {}).get("id", "")


def register_action_handlers(app) -> None:
    """Register all action handlers on the Bolt app."""

    # ── Root cause button ──────────────────────────────────────────────────
    @app.action(re.compile(r"^root_cause__(.+)$"))
    def handle_root_cause(ack, body, client, action):
        ack()
        metric_key = action["value"]
        channel_id = body["channel"]["id"]
        # Always use the root thread ts — handles both the original alert (no thread_ts)
        # and the "still at risk" follow-up message (which is itself a thread reply)
        thread_ts = body["message"].get("thread_ts") or body["message"]["ts"]

        device = _extract_device_from_body(body)
        article_mrkdwn = get_article(metric_key, device)
        blocks = build_article_blocks(article_mrkdwn, metric_key, is_replacement(metric_key))

        client.chat_postMessage(
            channel=channel_id,
            text=f"Here's how to address {METRIC_LABELS.get(metric_key, metric_key)}:",
            blocks=blocks,
            thread_ts=thread_ts,
        )

    # ── Self-service "Done, re-run diagnosis" ──────────────────────────────
    @app.action(re.compile(r"^done_selfservice__(.+)$"))
    def handle_done_selfservice(ack, body, client, action):
        ack()
        metric_key = action["value"]
        channel_id = body["channel"]["id"]
        username = _get_username(body)
        # Continue in the same thread as the original alert
        thread_ts = body["message"].get("thread_ts") or body["message"]["ts"]

        # Post loading message, then update it in-place once diagnosis returns
        loading = client.chat_postMessage(
            channel=channel_id,
            text="Running a fresh check on your laptop... ⏳",
            thread_ts=thread_ts,
        )
        loading_ts = loading["ts"]

        run_diagnosis(username)

        client.chat_update(
            channel=channel_id,
            ts=loading_ts,
            text=(
                "Your laptop is all clear! No risk factors detected. 🎉\n\n"
                "Great job taking action — you've kept your laptop running smoothly. "
                "Feel free to message me here if anything else comes up!"
            ),
        )

        # Update the original alert message to show resolved state
        _update_original_alert(body, client, resolved=True)

    # ── Replacement "Got it, I'll back up my data" ─────────────────────────
    @app.action(re.compile(r"^done_replacement__(.+)$"))
    def handle_done_replacement(ack, body, client, action):
        ack()
        channel_id = body["channel"]["id"]
        thread_ts = body["message"].get("thread_ts") or body["message"]["ts"]

        client.chat_postMessage(
            channel=channel_id,
            text=(
                "Thanks for taking this seriously! 💙 Backing up your data is the "
                "most important step right now.\n\n"
                "When you're ready to request a replacement, reach out to IT — "
                "they'll take great care of you. Message me here anytime if you need help!"
            ),
            thread_ts=thread_ts,
        )

        _update_original_alert(body, client, resolved=True)

    # ── Replacement process actions ────────────────────────────────────────
    @app.action(re.compile(r"^replacement_action__(.+)$"))
    def handle_replacement_action(ack, body, client, action):
        ack()
        article_key = action["action_id"].replace("replacement_action__", "")
        channel_id = body["channel"]["id"]
        thread_ts = body["message"].get("thread_ts") or body["message"]["ts"]

        article_mrkdwn = get_article(article_key)
        client.chat_postMessage(
            channel=channel_id,
            text=article_mrkdwn,
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": article_mrkdwn},
                }
            ],
            thread_ts=thread_ts,
        )

    # ── Snooze ─────────────────────────────────────────────────────────────
    @app.action("snooze")
    def handle_snooze(ack, body, client):
        ack()
        channel_id = body["channel"]["id"]
        message_ts = body["message"]["ts"]

        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text="Snoozed — I'll check back with you in an hour.",
            blocks=build_snoozed_blocks(),
        )

    # ── Opt out ────────────────────────────────────────────────────────────
    @app.action("opt_out")
    def handle_opt_out(ack, body, client):
        ack()
        channel_id = body["channel"]["id"]
        message_ts = body["message"]["ts"]

        # Update original alert to show unsubscribed state
        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text="You've unsubscribed from alerts like this.",
            blocks=build_opted_out_blocks(),
        )

        # Post follow-up asking for optional feedback
        client.chat_postMessage(
            channel=channel_id,
            text="Got it. You've unsubscribed from alerts like this.",
            blocks=build_opt_out_feedback_blocks(),
        )

    # ── Opt-out feedback ───────────────────────────────────────────────────
    @app.action(re.compile(r"^opt_out_reason__(.+)$"))
    def handle_opt_out_reason(ack, body, client, action):
        ack()
        reason = action["action_id"].replace("opt_out_reason__", "")
        username = _get_username(body)
        channel_id = body["channel"]["id"]
        message_ts = body["message"]["ts"]

        # Save opt-out with reason
        opt_outs = _load_opt_outs()
        opt_outs[username] = reason if reason != "skip" else "no_reason_given"
        _save_opt_outs(opt_outs)

        # Replace the feedback buttons with a thank-you
        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text="Thanks for letting us know!",
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Thanks for letting us know — that really helps us improve! 💙",
                    },
                }
            ],
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _update_original_alert(body: dict, client, resolved: bool) -> None:
    """
    Update the original alert message to preserve its content, remove action
    buttons, and append a small resolved/acknowledged footer.
    """
    channel_id = body["channel"]["id"]
    # When called after a thread reply button click (e.g. "Done"), body["message"]
    # is the article reply — use thread_ts to get back to the original alert.
    original_ts = body["message"].get("thread_ts") or body["message"]["ts"]

    # Fetch the original alert blocks so we can preserve them
    try:
        history = client.conversations_history(
            channel=channel_id,
            latest=original_ts,
            inclusive=True,
            limit=1,
        )
        msgs = history.get("messages", [])
        original_blocks = msgs[0].get("blocks", []) if msgs else []
        # Keep text/section/context/divider blocks — strip action buttons
        preserved = [b for b in original_blocks if b.get("type") != "actions"]
    except Exception:
        preserved = []

    # Append a small resolved footer
    preserved.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "✅ Resolved — see thread for details"}],
    })

    client.chat_update(
        channel=channel_id,
        ts=original_ts,
        text="Alert resolved.",
        blocks=preserved,
    )


def _extract_risk_labels_from_body(body: dict) -> list:
    """
    Parse the original alert blocks to extract risk display labels.
    Falls back to an empty list if parsing fails.
    """
    try:
        blocks = body["message"]["blocks"]
        for block in blocks:
            if block.get("block_id") == "root_cause_actions":
                return [el["text"]["text"] for el in block.get("elements", [])]
    except (KeyError, TypeError):
        pass
    return []


def _extract_device_from_body(body: dict) -> str:
    """
    Try to extract the device name from the original alert message text.
    Falls back to "your device".
    """
    try:
        text = body["message"]["blocks"][0]["text"]["text"]
        match = re.search(r"\*([\w\-]+)\*", text)
        if match:
            return match.group(1)
    except (KeyError, TypeError, IndexError):
        pass
    return "your device"
