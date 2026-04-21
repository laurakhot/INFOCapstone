"""
Button action handlers for the laptop health alert bot.

Action IDs handled here:
  rerun_diagnosis                    — user clicks "Re-run diagnosis" from the alert
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
    build_still_at_risk_blocks,
    build_snoozed_blocks,
    build_opted_out_blocks,
    build_opt_out_feedback_blocks,
    METRIC_LABELS,
    METRIC_BUTTON_LABELS,
)
from backend_client import run_diagnosis
from rag.articles import get_article, get_canvas_url, is_replacement

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
    return body.get("user", {}).get("username", "unknown")


def _get_user_id(body: dict) -> str:
    return body.get("user", {}).get("id", "")


def _run_diagnosis_and_reply(body: dict, client, channel_id: str, thread_ts: str, username: str) -> None:
    """Shared logic for re-run diagnosis: post loading msg, call backend, reply in thread."""
    loading = client.chat_postMessage(
        channel=channel_id,
        text="Running a fresh check on your laptop... ⏳",
        thread_ts=thread_ts,
    )
    loading_ts = loading["ts"]

    result = run_diagnosis(username)
    remaining_features = list(result.keys()) if result else []

    if remaining_features:
        canvas_entries = [
            (get_canvas_url(k), METRIC_BUTTON_LABELS.get(k, METRIC_LABELS.get(k, k)))
            for k in remaining_features
        ]
        still_at_risk_blocks = build_still_at_risk_blocks(remaining_features, canvas_entries)
        client.chat_update(
            channel=channel_id,
            ts=loading_ts,
            text="I'm still seeing some risk factors on your laptop.",
            blocks=still_at_risk_blocks,
        )
    else:
        client.chat_update(
            channel=channel_id,
            ts=loading_ts,
            text=(
                "Your laptop is all clear! No risk factors detected. 🎉\n\n"
                "Great job taking action — you've kept your laptop running smoothly. "
                "Feel free to message me here if anything else comes up!"
            ),
        )
        _update_original_alert(body, client, resolved=True)


def register_action_handlers(app) -> None:
    """Register all action handlers on the Bolt app."""

    # ── Re-run diagnosis (from initial alert) ─────────────────────────────
    @app.action("rerun_diagnosis")
    def handle_rerun_diagnosis(ack, body, client):
        ack()
        channel_id = body["channel"]["id"]
        username = _get_username(body)
        thread_ts = body["message"].get("thread_ts") or body["message"]["ts"]
        _run_diagnosis_and_reply(body, client, channel_id, thread_ts, username)

    # ── Self-service "Done, re-run diagnosis" ──────────────────────────────
    @app.action(re.compile(r"^done_selfservice__(.+)$"))
    def handle_done_selfservice(ack, body, client, action):
        ack()
        channel_id = body["channel"]["id"]
        username = _get_username(body)
        thread_ts = body["message"].get("thread_ts") or body["message"]["ts"]
        _run_diagnosis_and_reply(body, client, channel_id, thread_ts, username)

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

        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text="You've unsubscribed from alerts like this.",
            blocks=build_opted_out_blocks(),
        )

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

        opt_outs = _load_opt_outs()
        opt_outs[username] = reason if reason != "skip" else "no_reason_given"
        _save_opt_outs(opt_outs)

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
    Update the original alert message: strip action buttons, append resolved footer.
    """
    channel_id = body["channel"]["id"]
    original_ts = body["message"].get("thread_ts") or body["message"]["ts"]

    try:
        history = client.conversations_history(
            channel=channel_id,
            latest=original_ts,
            inclusive=True,
            limit=1,
        )
        msgs = history.get("messages", [])
        original_blocks = msgs[0].get("blocks", []) if msgs else []
        preserved = [b for b in original_blocks if b.get("type") != "actions"]
    except Exception:
        preserved = []

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


def _extract_device_from_body(body: dict) -> str:
    """
    Try to extract the device name from the original alert message text.
    Falls back to "your device".
    """
    try:
        text = body["message"]["blocks"][0]["text"]["text"]
        match = re.search(r"\*([\w\s\-]+)\*", text)
        if match:
            return match.group(1)
    except (KeyError, TypeError, IndexError):
        pass
    return "your device"
