"""
Block Kit message builders for laptop health alerts.
"""

from typing import Optional

METRIC_LABELS = {
    "avg_processor_time":     "Processor Time",
    "max_cpu_usage":          "Max CPU Usage",
    "avg_memory_utilization": "Average Memory Utilization",
    "avg_battery_health":     "Battery Health",
    "uptime_days":            "System Uptime",
    "p90_cpu_temp":           "CPU Temperature",
}

METRIC_BUTTON_LABELS = {
    "avg_processor_time":     "Check Processor Performance",
    "max_cpu_usage":          "Optimize CPU Usage",
    "avg_memory_utilization": "Reduce Memory Usage",
    "avg_battery_health":     "Check Battery Health",
    "uptime_days":            "Restart your Laptop",
    "p90_cpu_temp":           "Cool Down your Laptop",
}


def build_alert_blocks(
    first_name: str,
    device: str,
    canvas_entries: list,  # list of (canvas_url: Optional[str], label: str)
    has_replacement: bool = False,
) -> list:
    """
    Build Block Kit blocks for the initial alert DM.

    canvas_entries: list of (canvas_url, label) tuples for each flagged metric.
    Canvas URLs are included as mrkdwn bullet links; Slack auto-unfurls them as
    Canvas cards below the message.
    """
    if not canvas_entries:
        return []

    # Build bullet lines: linked if URL available, plain text fallback
    bullets = []
    for url, label in canvas_entries:
        if url:
            bullets.append(f"• <{url}|{label}>")
        else:
            bullets.append(f"• {label}")

    bullet_text = "\n".join(bullets)

    body = (
        f"Hey {first_name} 👋\n\n"
        f"Our system detected some risk factors that may cause your laptop to crash. 😵 "
        f"Here are guides based on what we found on your *{device}*:\n\n"
        f"{bullet_text}\n\n"
        f"Once you have followed the steps in the canvas, try re-running to see if your issue is fixed."
    )

    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": body},
        },
        {
            "type": "actions",
            "block_id": "alert_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Re-run diagnosis"},
                    "action_id": "rerun_diagnosis",
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Snooze for an hour"},
                    "action_id": "snooze",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Don't remind me again"},
                    "action_id": "opt_out",
                },
            ],
        },
    ]


def build_still_at_risk_blocks(remaining: dict, canvas_entries: list) -> list:
    """
    Follow-up shown after re-diagnosis when risk factors remain.

    canvas_entries: list of (canvas_url, label) for remaining metrics.
    """
    if not canvas_entries:
        return []

    bullets = []
    for url, label in canvas_entries:
        if url:
            bullets.append(f"• <{url}|{label}>")
        else:
            bullets.append(f"• {label}")

    bullet_text = "\n".join(bullets)

    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"I'm still seeing some risk factors. "
                    f"Here are the guides to work through:\n\n{bullet_text}\n\n"
                    f"Click \"re-run diagnosis\" again once you've gone through them."
                ),
            },
        },
        {
            "type": "actions",
            "block_id": "alert_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Re-run diagnosis"},
                    "action_id": "rerun_diagnosis",
                    "style": "primary",
                },
            ],
        },
    ]


def build_article_blocks(article_mrkdwn: str, metric_key: str, is_replacement: bool = False) -> list:
    """Build Block Kit blocks to display a KB article with a resolution button."""
    if is_replacement:
        button_text = "✅ Got it, I'll back up my data"
        action_id = f"done_replacement__{metric_key}"
    else:
        button_text = "✅ Done, re-run the diagnosis"
        action_id = f"done_selfservice__{metric_key}"

    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": article_mrkdwn},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Once you've gone through the steps, I can run a fresh check "
                    "to confirm your laptop is back to normal!"
                    if not is_replacement
                    else "When you're ready, let me know!"
                ),
            },
        },
        {
            "type": "actions",
            "block_id": f"resolution_actions__{metric_key}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": button_text},
                    "action_id": action_id,
                    "value": metric_key,
                    "style": "primary",
                }
            ],
        },
    ]


def build_acknowledged_blocks(risk_labels: list) -> list:
    """Replacement for the original alert after acknowledge."""
    labels_text = " · ".join(risk_labels)
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"✅ *Issue resolved*\n_{labels_text}_",
            },
        }
    ]


def build_snoozed_blocks() -> list:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "⏰ *Snoozed* — I'll check back with you in an hour. Take care 😊",
            },
        }
    ]


def build_opted_out_blocks() -> list:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "🔕 *You've unsubscribed from alerts like this.*",
            },
        }
    ]


def build_opt_out_feedback_blocks() -> list:
    """Ask for optional opt-out feedback."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Got it. You've unsubscribed from alerts like this. 👍\n\n"
                    "Just out of curiosity, what made you decide to opt out? "
                    "_(Feel free to skip — this is totally optional 😊)_"
                ),
            },
        },
        {
            "type": "actions",
            "block_id": "opt_out_feedback_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Seems like a false positive"},
                    "action_id": "opt_out_reason__false_positive",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Already replacing my laptop"},
                    "action_id": "opt_out_reason__replacing",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Getting too many alerts"},
                    "action_id": "opt_out_reason__too_many",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Skip"},
                    "action_id": "opt_out_reason__skip",
                },
            ],
        },
    ]
